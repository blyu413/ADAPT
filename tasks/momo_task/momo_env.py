"""MoMo Lumped Environment - extends velocity task with momentum-based disturbance observation.

This environment maintains an ideal robot (with original unrandomized parameters)
and uses a Momentum Observer (MoMo) for disturbance estimation. The MoMo tracks
changes in generalized momentum p = M(q) * qdot and compares against expected
dynamics from the ideal (unrandomized) model.

The MoMo disturbance estimate captures:
  - Ground reaction forces (GRF) from the domain-randomized actual robot
  - Model mismatch due to domain randomization (mass, COM, damping, friction)
  - External disturbances (wrench events)

Theory (De Luca & Mattone 2003):
    p_dot = beta + tau_v + tau_e
    where beta = C^T qdot - g (approximated as -qfrc_bias)
    The observer residual r converges to tau_e.
"""

from __future__ import annotations
from typing import Any
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, VecEnvStepReturn, types
from mjlab.utils.logging import print_info
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from prettytable import PrettyTable
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from tasks.momo_task.mdp.wrench_events import step_continuous_wrenches

_WRENCH_STATE_ATTRS = [
    "wrench_state_static_force",
    "wrench_state_static_torque",
    "wrench_state_impulse_force",
    "wrench_state_impulse_torque",
]
_HAND_WRENCH_STATE_ATTRS = ["wrench_state_hand_load"]


def _build_ko_vector(ko_root=3.0, ko_legs=3.0, ko_waist=3.0, ko_arms=3.0, nv=35):
    """Build G1 per-DOF gains in base, legs, waist, arms order."""
    K_O = [0.0] * nv
    for i in range(0, 6):
        K_O[i] = ko_root
    for i in range(6, 18):
        K_O[i] = ko_legs
    for i in range(18, 21):
        K_O[i] = ko_waist
    for i in range(21, nv):
        K_O[i] = ko_arms
    return K_O


class MoMoEnv(ManagerBasedRlEnv):
    """Velocity tracking environment with Momentum Observer (MoMo).

    This environment extends the base ManagerBasedRlEnv by maintaining an ideal
    robot simulation for disturbance estimation. The ideal robot:
    - Uses the original unrandomized model parameters
    - Has its state synchronized from the actual robot each step
    - Provides qfrc_bias and qfrc_actuator for the MoMo update

    The MoMo uses generalized momentum p = M(q)*qdot from the ideal robot and
    compares momentum changes against expected dynamics (beta + tau_v) to estimate
    external disturbances tau_e. Since the actual robot is domain-randomized and
    the ideal robot is not, the residual r captures both model mismatch and GRF
    — matching the deployment scenario where a nominal model is used.
    """

    _momo_ko: float | list[float] | None = None
    _momo_butter_order: int = 2
    _momo_butter_cutoff: float = 1.0
    _qpos_noise_range: float = 0.01
    _qvel_noise_range: float = 0.5
    _encoder_bias_range: float = 0.015

    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        device: str,
        render_mode: str | None = None,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        self._rank = rank
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self.extras = {}
        self.obs_buf = {}
        self._manual_reset_pending = torch.zeros(
            self.cfg.scene.num_envs, dtype=torch.bool, device=device
        )
        self.scene = Scene(self.cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=device,
        )
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)
        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=device, dtype=torch.long)
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(
                model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
            )
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt
        mj_model = self.sim.mj_model
        mj_data = self.sim.mj_data
        warp_device = wp.get_device(self.device)
        with wp.ScopedDevice(warp_device):
            self.ideal_wp_model = mjwarp.put_model(mj_model)
            self.ideal_data = mjwarp.put_data(
                mj_model,
                mj_data,
                nworld=self.num_envs,
                nconmax=self.sim.cfg.nconmax,
                njmax=self.sim.cfg.njmax,
            )
            self.ideal_qpos_view = wp.to_torch(self.ideal_data.qpos)
            self.ideal_qvel_view = wp.to_torch(self.ideal_data.qvel)
            self.ideal_ctrl_view = wp.to_torch(self.ideal_data.ctrl)
        from tasks.momo_task.mdp.momo_observer import VectorizedMoMo

        nv = self.sim.mj_model.nv
        nu = self.sim.mj_model.nu
        default_body_mass = self.sim.get_default_field("body_mass")
        gravity_z = self.sim.mj_model.opt.gravity[2]
        self.nominal_mg = float(default_body_mass.sum() * abs(gravity_z))
        step_dt = self.cfg.sim.mujoco.timestep * self.cfg.decimation
        ko = self._momo_ko if self._momo_ko is not None else _build_ko_vector(nv=nv)
        self.momo_observer = VectorizedMoMo(
            num_envs=self.num_envs,
            nv=nv,
            K_O=ko,
            dt=step_dt,
            device=self.device,
            nominal_mg=self.nominal_mg,
            butter_order=self._momo_butter_order,
            butter_cutoff=self._momo_butter_cutoff,
        )
        self.momo_observer.initialize_state()
        with wp.ScopedDevice(warp_device):
            self._momentum_buf = wp.zeros((self.num_envs, nv), dtype=wp.float32)
        self._momentum_torch = wp.to_torch(self._momentum_buf)
        self._nu = nu
        self._nq_base = self.sim.mj_model.nq - nu
        self._nv_base = nv - nu
        self._encoder_bias = torch.zeros(self.num_envs, nu, device=self.device)
        if self._encoder_bias_range > 0:
            self._encoder_bias.uniform_(-self._encoder_bias_range, self._encoder_bias_range)
        print(
            f"[MoMo Observer] Initialized with K_O={ko}, nv={nv}, butter={self._momo_butter_order}/{self._momo_butter_cutoff}Hz, noise=qpos±{self._qpos_noise_range}/qvel±{self._qvel_noise_range}/bias±{self._encoder_bias_range}."
        )
        self.load_managers()
        self.setup_manager_visualizers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.process_action(action.to(self.device))
        for decimation_step in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.ideal_qpos_view[:] = self.sim.data.qpos
            self.ideal_qvel_view[:] = self.sim.data.qvel
            step_continuous_wrenches(
                self,
                _WRENCH_STATE_ATTRS,
                "robot",
                ("torso_link",),
                clean_flag_attr="wrench_clean_env_flag",
                category_attr="wrench_category",
                enable_categories=(1, 4),
            )
            step_continuous_wrenches(
                self,
                _HAND_WRENCH_STATE_ATTRS,
                "robot",
                ("left_wrist_yaw_link", "right_wrist_yaw_link"),
                clean_flag_attr="wrench_clean_env_flag",
            )
            step_continuous_wrenches(
                self,
                ["wrench_state_hand_load_left"],
                "robot",
                ("left_wrist_yaw_link",),
                clean_flag_attr="wrench_clean_env_flag",
                category_attr="wrench_category",
                enable_categories=(2, 4),
            )
            step_continuous_wrenches(
                self,
                ["wrench_state_hand_load_right"],
                "robot",
                ("right_wrist_yaw_link",),
                clean_flag_attr="wrench_clean_env_flag",
                category_attr="wrench_category",
                enable_categories=(3, 4),
            )
            self.ideal_ctrl_view[:] = self.sim.data.ctrl
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
            self.metrics_manager.compute_substep()
        self.ideal_qpos_view[:] = self.sim.data.qpos
        self.ideal_qvel_view[:] = self.sim.data.qvel
        self.ideal_ctrl_view[:] = self.sim.data.ctrl
        if self._qpos_noise_range > 0 or self._encoder_bias_range > 0:
            noise_pos = torch.zeros_like(self._encoder_bias)
            if self._qpos_noise_range > 0:
                noise_pos.uniform_(-self._qpos_noise_range, self._qpos_noise_range)
            self.ideal_qpos_view[:, self._nq_base :] += self._encoder_bias + noise_pos
        if self._qvel_noise_range > 0:
            noise_vel = torch.zeros(self.num_envs, self._nu, device=self.device)
            noise_vel.uniform_(-self._qvel_noise_range, self._qvel_noise_range)
            self.ideal_qvel_view[:, self._nv_base :] += noise_vel
        with wp.ScopedDevice(self.sim.wp_device):
            mjwarp.forward(self.ideal_wp_model, self.ideal_data)
            mjwarp.mul_m(
                self.ideal_wp_model, self.ideal_data, self._momentum_buf, self.ideal_data.qvel
            )
        beta = -wp.to_torch(self.ideal_data.qfrc_bias)
        tau_v = wp.to_torch(self.ideal_data.qfrc_actuator)
        tau_passive = wp.to_torch(self.ideal_data.qfrc_passive)
        self.momo_observer.update(self._momentum_torch, beta, tau_v, tau_passive)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.momo_observer.reset(reset_env_ids)
            if self._encoder_bias_range > 0:
                self._encoder_bias[reset_env_ids].uniform_(
                    -self._encoder_bias_range, self._encoder_bias_range
                )
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.ideal_qpos_view[reset_env_ids] = self.sim.data.qpos[reset_env_ids]
            self.ideal_qvel_view[reset_env_ids] = self.sim.data.qvel[reset_env_ids]
            self.ideal_ctrl_view[reset_env_ids] = self.sim.data.ctrl[reset_env_ids]
            with wp.ScopedDevice(self.sim.wp_device):
                mjwarp.forward(self.ideal_wp_model, self.ideal_data)
                mjwarp.mul_m(
                    self.ideal_wp_model, self.ideal_data, self._momentum_buf, self.ideal_data.qvel
                )
            self.momo_observer.p_prev[reset_env_ids] = self._momentum_torch[reset_env_ids]
        self.sim.forward()
        self.command_manager.compute(dt=self.step_dt)
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[types.VecEnvObs, dict]:
        """Reset the environment and ideal robot simulation.

        Args:
          seed: Random seed for the environment.
          env_ids: Environment IDs to reset. If None, resets all environments.
          options: Additional options (unused).

        Returns:
          Tuple of (observations, info dict).
        """
        del options
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if seed is not None:
            self.seed(seed)
        self._reset_idx(env_ids)
        self.momo_observer.reset(env_ids)
        if self._encoder_bias_range > 0:
            self._encoder_bias[env_ids].uniform_(
                -self._encoder_bias_range, self._encoder_bias_range
            )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.ideal_qpos_view[env_ids] = self.sim.data.qpos[env_ids]
        self.ideal_qvel_view[env_ids] = self.sim.data.qvel[env_ids]
        self.ideal_ctrl_view[env_ids] = self.sim.data.ctrl[env_ids]
        with wp.ScopedDevice(self.sim.wp_device):
            mjwarp.forward(self.ideal_wp_model, self.ideal_data)
            mjwarp.mul_m(
                self.ideal_wp_model, self.ideal_data, self._momentum_buf, self.ideal_data.qvel
            )
        self.momo_observer.p_prev[env_ids] = self._momentum_torch[env_ids]
        self.command_manager.compute(dt=0.0)
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return (self.obs_buf, self.extras)

    def setup_manager_visualizers(self) -> None:
        """Setup visualizers for managers."""
        self.manager_visualizers = {}
        if getattr(self.command_manager, "active_terms", None):
            self.manager_visualizers["command_manager"] = self.command_manager
