from __future__ import annotations
from typing import TYPE_CHECKING

import math

import torch

from mjlab.entity import Entity

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import momo_leg_effort_scaled_envelope

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _soft_margin_excess(
    value: torch.Tensor,
    threshold: float,
    margin: float,
) -> torch.Tensor:
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    return torch.clamp((value - threshold) / margin, min=0.0)


def _moving_command_mask(
    env: "ManagerBasedRlEnv",
    command_name: str,
    command_threshold: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if command_threshold <= 0.0:
        return torch.ones(env.num_envs, device=env.device, dtype=dtype)
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    lin_moving = torch.norm(command[:, :2], dim=-1) > command_threshold
    yaw_moving = torch.abs(command[:, 2]) > command_threshold
    return (lin_moving | yaw_moving).to(dtype=dtype)


class _momo_leg_envelope_history:
    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.history_length = int(cfg.params["history_length"])
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive, got {self.history_length}")
        self.eps = float(cfg.params.get("eps", 1e-6))
        self.history = torch.zeros(
            env.num_envs, self.history_length, device=env.device, dtype=torch.float
        )
        self.cursor = 0
        self.last_step = -1

    def _append_current(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        step = int(env.common_step_counter)
        if self.last_step != step:
            momo = momo_leg_effort_scaled_envelope(
                env,
                asset_cfg=self.asset_cfg,
                eps=self.eps,
            )
            leg_env_max = torch.max(momo[:, -2:], dim=-1).values
            self.history[:, self.cursor] = leg_env_max
            self.cursor = (self.cursor + 1) % self.history_length
            self.last_step = step
        return self.history

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0


class momo_leg_cvar_tail(_momo_leg_envelope_history):
    """Rolling CVaR penalty for high-tail leg MoMo effort envelope."""

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        asset_cfg: SceneEntityCfg,
        history_length: int,
        threshold: float,
        margin: float,
        command_name: str,
        command_threshold: float = 0.05,
        quantile: float = 0.9,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        del asset_cfg, history_length, eps  # Cached at init.
        if not 0.0 <= quantile < 1.0:
            raise ValueError(f"quantile must be in [0, 1), got {quantile}")
        history = self._append_current(env)
        topk = max(1, math.ceil(self.history_length * (1.0 - quantile)))
        cvar = torch.topk(history, k=topk, dim=-1).values.mean(dim=-1)
        penalty = _soft_margin_excess(cvar, threshold=threshold, margin=margin)
        moving = _moving_command_mask(env, command_name, command_threshold, penalty.dtype)
        return moving * penalty


class momo_leg_peak_guard(_momo_leg_envelope_history):
    """Rolling max penalty for extreme leg MoMo effort envelope peaks."""

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        asset_cfg: SceneEntityCfg,
        history_length: int,
        threshold: float,
        margin: float,
        command_name: str,
        command_threshold: float = 0.05,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        del asset_cfg, history_length, eps  # Cached at init.
        peak = torch.max(self._append_current(env), dim=-1).values
        penalty = _soft_margin_excess(peak, threshold=threshold, margin=margin)
        moving = _moving_command_mask(env, command_name, command_threshold, penalty.dtype)
        return moving * penalty


def momo_leg_softstep_reward_schedule(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    stages: list[dict],
) -> dict[str, torch.Tensor]:
    del env_ids
    if len(stages) == 0:
        return {}

    step = env.common_step_counter
    active_stage = stages[0]
    for stage in stages:
        if step >= stage["step"]:
            active_stage = stage

    updates = (
        ("momo_leg_cvar_tail", "cvar"),
        ("momo_leg_peak_guard", "peak"),
    )
    log: dict[str, torch.Tensor] = {}
    for reward_name, prefix in updates:
        term_cfg = env.reward_manager.get_term_cfg(reward_name)
        term_cfg.weight = float(active_stage[f"{prefix}_weight"])
        term_cfg.params["threshold"] = float(active_stage[f"{prefix}_threshold"])
        term_cfg.params["margin"] = float(active_stage[f"{prefix}_margin"])

        log[f"{reward_name}_weight"] = torch.tensor(term_cfg.weight, device=env.device)
        log[f"{reward_name}_threshold"] = torch.tensor(
            term_cfg.params["threshold"],
            device=env.device,
        )
        log[f"{reward_name}_margin"] = torch.tensor(
            term_cfg.params["margin"],
            device=env.device,
        )

    return log


def track_linear_velocity(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track xy linear velocity in the yaw-aligned frame, without a coupled vz² penalty."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    v_w_xy = asset.data.root_link_lin_vel_w[:, :2]
    yaw = asset.data.heading_w
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    vx_y = cos_y * v_w_xy[:, 0] + sin_y * v_w_xy[:, 1]
    vy_y = -sin_y * v_w_xy[:, 0] + cos_y * v_w_xy[:, 1]

    xy_error = (command[:, 0] - vx_y) ** 2 + (command[:, 1] - vy_y) ** 2
    return torch.exp(-xy_error / std**2)


def track_angular_velocity(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track ωz using yaw error only, without a coupled ωxy² penalty."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-z_error / std**2)


def lin_vel_z_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Standalone vz² penalty replacing the term coupled into mjlab's track_linear_velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])
