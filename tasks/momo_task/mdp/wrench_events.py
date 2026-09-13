from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

RangeSpec = tuple[float, float] | dict[str, tuple[float, float]]


class WrenchState:
    """Per-environment wrench state: force/torque buffers, cycle timers, param ranges."""

    def __init__(
        self,
        force_range: RangeSpec = (0.0, 0.0),
        torque_range: RangeSpec = (0.0, 0.0),
        duration_range_s: tuple[float, float] = (5.0, 10.0),
        idle_range_s: tuple[float, float] = (4.0, 7.0),
    ) -> None:
        self.force_range: RangeSpec = force_range
        self.torque_range: RangeSpec = torque_range
        self.duration_range_s: tuple[float, float] = duration_range_s
        self.idle_range_s: tuple[float, float] = idle_range_s

        # Allocated in initialize()
        self.forces: torch.Tensor | None = None  # (num_envs, num_bodies, 3)
        self.torques: torch.Tensor | None = None  # (num_envs, num_bodies, 3)
        self.remaining_force: torch.Tensor | None = None  # (num_envs,) int64
        self.remaining_idle: torch.Tensor | None = None  # (num_envs,) int64

    def initialize(self, num_envs: int, num_bodies: int, device: str) -> None:
        """Allocate tensor buffers."""
        self.forces = torch.zeros((num_envs, num_bodies, 3), device=device)
        self.torques = torch.zeros((num_envs, num_bodies, 3), device=device)
        self.remaining_force = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.remaining_idle = torch.zeros(num_envs, dtype=torch.long, device=device)


def _sample_range(
    range_spec: RangeSpec,
    num_envs: int,
    num_bodies: int,
    device: str,
) -> torch.Tensor:
    """Sample (num_envs, num_bodies, 3) uniformly from a RangeSpec.

    Raises AssertionError for any (lo, hi) pair where lo > hi.
    """
    if isinstance(range_spec, dict):
        lo = torch.zeros(3, device=device)
        hi = torch.zeros(3, device=device)
        for axis_name, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
            if axis_name in range_spec:
                lo_val, hi_val = range_spec[axis_name]
                assert lo_val <= hi_val, (
                    f"range_spec axis '{axis_name}': min={lo_val} > max={hi_val}"
                )
                lo[axis_idx] = lo_val
                hi[axis_idx] = hi_val
    else:
        lo_val, hi_val = range_spec
        assert lo_val <= hi_val, f"range_spec: min={lo_val} > max={hi_val}"
        lo = torch.full((3,), lo_val, device=device)
        hi = torch.full((3,), hi_val, device=device)
    return torch.rand((num_envs, num_bodies, 3), device=device).mul_(hi - lo).add_(lo)


def wrench_event_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    state_attr: str,
    asset_cfg: SceneEntityCfg,
    force_stages: list[dict] | None = None,
    torque_stages: list[dict] | None = None,
    duration_stages: list[dict] | None = None,
    idle_stages: list[dict] | None = None,
) -> dict[str, torch.Tensor]:
    """Update wrench parameters from curriculum stages; create the WrenchState on first call.

    Called at every _reset_idx in both train and play modes.  Curriculum
    values always take priority over the WrenchState defaults.

    Args:
        env: The RL environment.
        env_ids: Unused; curriculum applies globally.
        state_attr: Attribute name for the WrenchState on ``env``.
        asset_cfg: SceneEntityCfg for the wrench target body.  Resolved
            lazily here if CurriculumManager has not already done so.
        force_stages:    [{"step": int, "range": RangeSpec}, ...]
        torque_stages:   [{"step": int, "range": RangeSpec}, ...]
        duration_stages: [{"step": int, "range": (min_s, max_s)}, ...]
        idle_stages:     [{"step": int, "range": (min_s, max_s)}, ...]

    Returns:
        Loggable dict with current force/torque range values.
    """
    del env_ids  # Curriculum is applied globally; env_ids not used.

    if not hasattr(env, state_attr):
        if asset_cfg.body_ids is None:
            asset_cfg.resolve(env.scene)
        state = WrenchState()
        state.initialize(env.num_envs, len(asset_cfg.body_ids), env.device)
        setattr(env, state_attr, state)

    state: WrenchState = getattr(env, state_attr)
    step = env.common_step_counter

    for stages, attr in (
        (force_stages, "force_range"),
        (torque_stages, "torque_range"),
        (duration_stages, "duration_range_s"),
        (idle_stages, "idle_range_s"),
    ):
        if stages is not None:
            for s in stages:
                if step >= s["step"]:
                    setattr(state, attr, s["range"])

    return _log_ranges(state.force_range, state.torque_range)


def step_continuous_wrenches(
    env: ManagerBasedRlEnv,
    wrench_state_attrs: list[str],
    asset_name: str,
    body_names: tuple[str, ...],
    clean_flag_attr: str | None = None,
    category_attr: str | None = None,
    enable_categories: tuple[int, ...] | None = None,
    body_enable_categories: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """Apply wrenches and advance the force/idle cycle for all WrenchStates.

    Manages the full per-environment cycle without any EventManager timer
    manipulation:
        idle (idle_range_s) → force active (duration_range_s) → idle → …

    All env-level operations are vectorized; the small fixed list of
    ``wrench_state_attrs`` is the only Python loop.

    Args:
        env: The RL environment.
        wrench_state_attrs: WrenchState attribute names on ``env``.
        asset_name: Asset name in the scene (e.g. ``"robot"``).
        body_names: Body names to receive the wrenches.
        clean_flag_attr: Optional attribute name for a per-env bool tensor on
            ``env``.  Where True, all wrench output is masked to zero before
            being written to sim.  If the attribute is absent (i.e. no clean
            curriculum configured for this task), masking is a no-op.
        category_attr: Optional attribute name for a per-env long tensor on
            ``env`` holding category labels.  Combined with ``enable_categories``
            for per-env selective output.
        enable_categories: Tuple of category labels for which this call's
            wrenches stay active; envs whose category is NOT in this tuple have
            their forces/torques masked to zero.  Both kwargs must be provided
            and the attr must exist on ``env`` for masking to take effect.
        body_enable_categories: Per-body category labels that keep each body's
            wrench active. Used when one synchronized wrench state targets
            multiple bodies but the active bodies depend on the category.
    """
    # Early return: if none of this call's WrenchStates exist on env, do nothing.
    # Otherwise we would write a zero-wrench to body_names and clobber any
    # earlier call that targeted the same body in this substep.
    if not any(hasattr(env, s) for s in wrench_state_attrs):
        return

    cache_attr = f"_wrench_asset_cfg_{asset_name}_{'_'.join(body_names)}"
    if not hasattr(env, cache_attr):
        cfg = SceneEntityCfg(asset_name, body_names=body_names)
        cfg.resolve(env.scene)
        setattr(env, cache_attr, cfg)
    asset_cfg = getattr(env, cache_attr)

    asset = env.scene[asset_name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1
    device = env.device

    combined_forces = torch.zeros((env.num_envs, num_bodies, 3), device=device)
    combined_torques = torch.zeros((env.num_envs, num_bodies, 3), device=device)

    for state_attr in wrench_state_attrs:
        if not hasattr(env, state_attr):
            continue  # WrenchState not created (no wrench curriculum configured)
        state: WrenchState = getattr(env, state_attr)

        # 1. Accumulate forces for currently-active envs.
        active = state.remaining_force > 0
        mask = active.float().unsqueeze(-1).unsqueeze(-1)
        combined_forces.add_(state.forces * mask)
        combined_torques.add_(state.torques * mask)

        # 2. Advance force countdown.
        state.remaining_force.sub_(1).clamp_(min=0)

        # 3. Force just ended → sample idle duration for those envs.
        force_just_ended = active & (state.remaining_force == 0)
        if force_just_ended.any():
            ids = force_just_ended.nonzero(as_tuple=False).squeeze(-1)
            lo, hi = state.idle_range_s
            state.remaining_idle[ids] = (
                ((torch.rand(len(ids), device=device) * (hi - lo) + lo) / env.physics_dt)
                .long()
                .clamp_(min=0)
            )

        # 4. Advance idle countdown.
        state.remaining_idle.sub_(1).clamp_(min=0)

        # 5. Idle ended → trigger new force (only for envs not in force phase).
        idle_ended = ~active & (state.remaining_idle == 0)
        if idle_ended.any():
            ids = idle_ended.nonzero(as_tuple=False).squeeze(-1)
            n = len(ids)
            state.forces[ids] = _sample_range(state.force_range, n, num_bodies, device)
            state.torques[ids] = _sample_range(state.torque_range, n, num_bodies, device)
            lo, hi = state.duration_range_s
            state.remaining_force[ids] = (
                ((torch.rand(n, device=device) * (hi - lo) + lo) / env.physics_dt)
                .long()
                .clamp_(min=1)
            )

    if clean_flag_attr is not None and hasattr(env, clean_flag_attr):
        clean_flag: torch.Tensor = getattr(env, clean_flag_attr)
        # clean_flag: (num_envs,) bool. Zero out wrenches where True.
        keep_mask = (~clean_flag).to(combined_forces.dtype).unsqueeze(-1).unsqueeze(-1)
        combined_forces.mul_(keep_mask)
        combined_torques.mul_(keep_mask)

    if category_attr is not None and enable_categories is not None and hasattr(env, category_attr):
        cat: torch.Tensor = getattr(env, category_attr)
        enabled = torch.as_tensor(enable_categories, device=device, dtype=cat.dtype)
        keep = torch.isin(cat, enabled)
        cat_mask = keep.to(combined_forces.dtype).unsqueeze(-1).unsqueeze(-1)
        combined_forces.mul_(cat_mask)
        combined_torques.mul_(cat_mask)

    if (
        category_attr is not None
        and body_enable_categories is not None
        and hasattr(env, category_attr)
    ):
        cat = getattr(env, category_attr)
        for body_idx, categories in enumerate(body_enable_categories):
            enabled = torch.as_tensor(categories, device=device, dtype=cat.dtype)
            keep = torch.isin(cat, enabled).to(combined_forces.dtype).unsqueeze(-1)
            combined_forces[:, body_idx].mul_(keep)
            combined_torques[:, body_idx].mul_(keep)

    asset.write_external_wrench_to_sim(
        combined_forces,
        combined_torques,
        env_ids=None,
        body_ids=body_ids,
    )


def reset_wrench_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    state_attr: str,
) -> None:
    """Reset event for a single WrenchState.

    Registered as ``EventTermCfg(mode="reset")`` so the EventManager calls
    this whenever environments reset.  The curriculum has already run by
    this point (``curriculum_manager.compute`` precedes ``event_manager.apply``
    in ``_reset_idx``), so ``state.idle_range_s`` reflects the current
    curriculum stage.

    Args:
        env: The RL environment.
        env_ids: Environment indices being reset.
        state_attr: Attribute name for the WrenchState on ``env``.
    """
    state: WrenchState = getattr(env, state_attr)
    n = len(env_ids)
    state.forces[env_ids] = 0.0
    state.torques[env_ids] = 0.0
    state.remaining_force[env_ids] = 0
    lo, hi = state.idle_range_s
    state.remaining_idle[env_ids] = (
        ((torch.rand(n, device=env.device) * (hi - lo) + lo) / env.physics_dt).long().clamp_(min=0)
    )


def _log_ranges(
    force_range: RangeSpec,
    torque_range: RangeSpec,
) -> dict[str, torch.Tensor]:
    """Build a loggable dict from force/torque range specs."""
    result: dict[str, torch.Tensor] = {}
    for prefix, rng in (("force", force_range), ("torque", torque_range)):
        if isinstance(rng, dict):
            for ax in ("x", "y", "z"):
                if ax in rng:
                    result[f"{prefix}_{ax}_min"] = torch.tensor(rng[ax][0])
                    result[f"{prefix}_{ax}_max"] = torch.tensor(rng[ax][1])
        else:
            result[f"{prefix}_min"] = torch.tensor(rng[0])
            result[f"{prefix}_max"] = torch.tensor(rng[1])
    return result


def reset_clean_env_flag(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    flag_attr: str,
    clean_prob_attr: str,
) -> None:
    """Reset event: Bernoulli re-sample the clean flag for resetting envs.

    Reads current clean probability from ``env.<clean_prob_attr>`` (set by
    ``clean_env_curriculum``).  Allocates the flag tensor on first call.
    """
    if not hasattr(env, flag_attr):
        setattr(
            env,
            flag_attr,
            torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            ),
        )
    flag: torch.Tensor = getattr(env, flag_attr)
    prob = float(getattr(env, clean_prob_attr, 0.0))
    flag[env_ids] = torch.rand(len(env_ids), device=env.device) < prob


def clean_env_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    clean_prob_attr: str,
    prob_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Curriculum: update the clean probability scalar on env based on training step.

    Stored on ``env.<clean_prob_attr>``; ``reset_clean_env_flag`` reads it
    each reset to Bernoulli-sample new clean flags.
    """
    del env_ids  # global curriculum
    step = env.common_step_counter
    prob = 0.0
    for s in prob_stages:
        if step >= s["step"]:
            prob = float(s["value"])
    setattr(env, clean_prob_attr, prob)
    return {"clean_prob": torch.tensor(prob)}


def force_cmd_zero_in_settle_window(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_term_name: str,
    settle_duration_s: float,
) -> None:
    """Step event: zero vel command for envs whose episode_length is below the settle window.

    Runs each control step (mode="step") between command_manager.compute() and
    obs_manager.compute(), so the command override is reflected in the policy
    observation for that step.
    """
    del env_ids  # apply to all envs
    cmd_term = env.command_manager.get_term(command_term_name)
    in_settle = (env.episode_length_buf.float() * env.step_dt) < settle_duration_s
    if in_settle.any():
        cmd_term.vel_command_b[in_settle] = 0.0
        cmd_term.vel_command_w[in_settle] = 0.0
