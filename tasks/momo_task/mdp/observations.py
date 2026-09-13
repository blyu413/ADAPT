from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.momo_task.momo_env import MoMoEnv

import torch

from mjlab.entity import Entity

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

LEG_EFFORT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

_DEFAULT_LEG_EFFORT_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=LEG_EFFORT_JOINT_NAMES,
    preserve_order=True,
)


def momo_dist_localframe(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return complete MoMo disturbance with base in body frame and joints in joint space.

    Combines body-frame base disturbance with joint disturbance to provide the full
    disturbance vector for the policy. The base disturbance is transformed to robot
    local coordinates (forward/left/up), while joint disturbances remain in joint space.

    Returns:
      Complete MoMo disturbance estimate
      Shape: (num_envs, 6 + num_joints) - [base_body_frame(6), joint_disturbance(N)]

      Base components (body frame):
        - [0:3]: Linear disturbance [forward, left, up]
        - [3:6]: Angular disturbance [roll, pitch, yaw]

      Joint components (joint space):
        - [6:]: Joint disturbances
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get full MoMo disturbance estimate (world frame)
    r_world = env.momo_observer.r_filtered

    # Extract base disturbance (first 6 DOFs in world frame)
    lin_dist_world = r_world[
        :, 0:3
    ].clone()  # (num_envs, 3) — clone to avoid modifying observer state
    lin_dist_world[:, 2] -= env.nominal_mg  # subtract m·g baseline before rotation
    ang_dist_world = r_world[:, 3:6]  # (num_envs, 3)

    # Extract joint disturbance (already in joint space, no transformation needed)
    joint_v_adr = asset.indexing.joint_v_adr[asset_cfg.joint_ids]
    joint_dist = r_world[:, joint_v_adr]  # (num_envs, num_joints)

    # Get robot base quaternion (w, x, y, z format)
    base_quat_w = asset.data.root_link_quat_w  # (num_envs, 4)

    # Transform base disturbance from world frame to body frame
    lin_dist_body = quat_apply_inverse(base_quat_w, lin_dist_world)
    ang_dist_body = quat_apply_inverse(base_quat_w, ang_dist_world)

    # Concatenate: [base_body_frame(6), joint_space(N)]
    return torch.cat([lin_dist_body, ang_dist_body, joint_dist], dim=1)


def _momo_joint_ids(
    env: MoMoEnv,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if isinstance(asset_cfg.joint_ids, slice):
        return torch.arange(
            asset.num_joints,
            device=env.device,
            dtype=torch.long,
        )[asset_cfg.joint_ids]
    return torch.as_tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)


def _momo_effort_scale_ctrl_ids(
    env: MoMoEnv,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    joint_ids = _momo_joint_ids(env, asset, asset_cfg)
    cache = getattr(env, "_momo_effort_scale_ctrl_ids_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_momo_effort_scale_ctrl_ids_cache", cache)

    cache_key = (asset_cfg.name, tuple(int(i) for i in joint_ids.cpu().tolist()))
    if cache_key in cache:
        return cache[cache_key]

    ctrl_ids = asset.indexing.ctrl_ids.to(device=env.device, dtype=torch.long)
    if ctrl_ids.numel() == 0:
        raise ValueError("momo_dist_effort_scaled_localframe requires joint actuators")

    actuator_joint_ids = torch.as_tensor(
        env.sim.mj_model.actuator_trnid[:, 0],
        device=env.device,
        dtype=torch.long,
    )[ctrl_ids]
    joint_global_ids = asset.indexing.joint_ids[joint_ids].to(
        device=env.device,
        dtype=torch.long,
    )
    matches = joint_global_ids.unsqueeze(1) == actuator_joint_ids.unsqueeze(0)
    match_count = matches.sum(dim=1)
    if not torch.all(match_count == 1):
        bad_joint_ids = joint_ids[match_count != 1].cpu().tolist()
        bad_names = [asset.joint_names[int(i)] for i in bad_joint_ids]
        raise ValueError(
            f"Could not align exactly one actuator for MOMO effort-scaled joints: {bad_names}"
        )

    matched_ctrl_ids = ctrl_ids[matches.to(torch.long).argmax(dim=1)]
    cache[cache_key] = matched_ctrl_ids
    return matched_ctrl_ids


def momo_dist_effort_scaled_localframe(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    l_ref: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return local-frame MoMo residual divided by fixed physical scales.

    Layout is unchanged from ``momo_dist_localframe``:
    [base force 3D, base torque 3D, 29 joint residuals]. Only the numeric
    scale changes:
      base force  / (m*g)
      base torque / (m*g*l_ref)
      joint       / actuator effort limit
    """
    if l_ref <= 0.0:
        raise ValueError(f"l_ref must be positive, got {l_ref}")

    raw = momo_dist_localframe(env, asset_cfg=asset_cfg)
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = _momo_effort_scale_ctrl_ids(env, asset, asset_cfg)

    force_range = env.sim.model.actuator_forcerange[:, ctrl_ids, :]
    effort_limits = torch.max(torch.abs(force_range), dim=-1).values.to(dtype=raw.dtype)
    if torch.any(effort_limits <= eps):
        joint_ids = _momo_joint_ids(env, asset, asset_cfg)
        zero_joint_ids = joint_ids[torch.any(effort_limits <= eps, dim=0)].cpu().tolist()
        zero_names = [asset.joint_names[int(i)] for i in zero_joint_ids]
        raise ValueError(
            "MOMO effort-scaled observation requires positive actuator forcerange "
            f"for all joints, got non-positive limits for {zero_names}"
        )

    base_force_scale = torch.full(
        (env.num_envs, 3),
        float(env.nominal_mg),
        device=env.device,
        dtype=raw.dtype,
    )
    base_torque_scale = torch.full(
        (env.num_envs, 3),
        float(env.nominal_mg) * float(l_ref),
        device=env.device,
        dtype=raw.dtype,
    )
    scale = torch.cat([base_force_scale, base_torque_scale, effort_limits], dim=1)
    if raw.shape[1] != scale.shape[1]:
        raise ValueError(
            "MOMO effort-scaled observation scale shape mismatch: "
            f"raw={raw.shape[1]}, scale={scale.shape[1]}"
        )
    return raw / torch.clamp(scale, min=eps)


def momo_joint_effort_scaled(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return selected MoMo joint residuals divided by actuator effort limits."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = _momo_joint_ids(env, asset, asset_cfg)
    joint_v_adr = asset.indexing.joint_v_adr[joint_ids]
    raw = env.momo_observer.r_filtered[:, joint_v_adr]
    ctrl_ids = _momo_effort_scale_ctrl_ids(env, asset, asset_cfg)

    force_range = env.sim.model.actuator_forcerange[:, ctrl_ids, :]
    effort_limits = torch.max(torch.abs(force_range), dim=-1).values.to(dtype=raw.dtype)
    if torch.any(effort_limits <= eps):
        zero_joint_ids = joint_ids[torch.any(effort_limits <= eps, dim=0)].cpu().tolist()
        zero_names = [asset.joint_names[int(i)] for i in zero_joint_ids]
        raise ValueError(
            "MOMO leg effort-scaled observation requires positive actuator forcerange "
            f"for all joints, got non-positive limits for {zero_names}"
        )

    return raw / torch.clamp(effort_limits, min=eps)


def momo_leg_effort_scaled_envelope(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_LEG_EFFORT_ASSET_CFG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return 12 leg effort-scaled MoMo residuals plus left/right RMS envelopes."""
    leg_effort = momo_joint_effort_scaled(env, asset_cfg=asset_cfg, eps=eps)
    if leg_effort.shape[1] != 12:
        raise ValueError(
            "momo_leg_effort_scaled_envelope expects exactly 12 leg joints, "
            f"got {leg_effort.shape[1]}"
        )

    leg_effort_by_side = leg_effort.reshape(env.num_envs, 2, 6)
    leg_effort_env = torch.sqrt(torch.mean(leg_effort_by_side.square(), dim=-1))
    return torch.cat([leg_effort, leg_effort_env], dim=1)


def zero_momo_leg_effort_scaled_envelope(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_LEG_EFFORT_ASSET_CFG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a zero tensor with the same layout as leg MoMo effort envelope."""
    del asset_cfg, eps
    dim = len(LEG_EFFORT_JOINT_NAMES) + 2
    return torch.zeros(env.num_envs, dim, device=env.device)


def shared_joint_pos_rel(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint pos (relative to default) read from MoMoEnv's noisy ideal_qpos.

    The same noisy tensor that MOMO integrates as its sensor input, so actor
    and observer see one encoder reading (real-robot semantics). Requires
    MoMoEnv with noise+bias already injected into ideal_qpos_view at step end.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return env.ideal_qpos_view[:, env._nq_base :] - asset.data.default_joint_pos


def shared_joint_vel_rel(
    env: MoMoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint vel (relative to default) read from MoMoEnv's noisy ideal_qvel.

    Shares the noise realization with MOMO (see shared_joint_pos_rel).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return env.ideal_qvel_view[:, env._nv_base :] - asset.data.default_joint_vel


def foot_height(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
