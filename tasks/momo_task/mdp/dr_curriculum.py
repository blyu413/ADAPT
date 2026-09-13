from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_VERIFY_FIELDS = [
    ("body_mass", "body_mass", "body mass"),
    ("joint_damping", "dof_damping", "joint damping"),
    ("dof_armature", "dof_armature", "armature"),
    ("pd_gains", "actuator_gainprm", "Kp (gainprm[:,0])"),
    ("joint_friction", "dof_frictionloss", "joint friction"),
]


def _verify_dr_active(env: ManagerBasedRlEnv, step: int, stages: list[dict]):
    """Print actual model parameter statistics at stage transitions.

    Compares current sim.model values against defaults to confirm DR is active.
    Only prints when the curriculum stage changes (not every step).
    """
    # Determine current stage step (highest step threshold that has been reached)
    current_stage_step = 0
    for s in stages:
        if step >= s["step"]:
            current_stage_step = max(current_stage_step, s["step"])

    if current_stage_step == env._dr_curriculum_last_stage_step:
        return  # No stage change, skip

    env._dr_curriculum_last_stage_step = current_stage_step
    iter_num = step // 24  # approximate iteration number

    print(f"\n[DR Curriculum] Stage transition at step={step} (~iter {iter_num})")
    print(
        f"  {'Parameter':<25s} {'Default mean':>12s} {'Current mean':>12s} "
        f"{'Current std':>12s} {'Min':>10s} {'Max':>10s}"
    )
    print("  " + "-" * 75)

    for _, field_name, desc in _VERIFY_FIELDS:
        try:
            default = env.sim.get_default_field(field_name)
            current = getattr(env.sim.model, field_name)

            if field_name == "actuator_gainprm":
                # gainprm is (num_envs, nu, 10), we want [:, :, 0] = Kp
                default_vals = default[:, 0]  # (nu,)
                current_vals = current[:, :, 0]  # (num_envs, nu)
            else:
                default_vals = default  # (N,) or (num_envs, N)
                current_vals = current  # (num_envs, N)

            # Compute ratio to default
            if current_vals.dim() > default_vals.dim():
                ratio = current_vals / (default_vals.unsqueeze(0) + 1e-10)
            else:
                ratio = current_vals / (default_vals + 1e-10)

            r_mean = ratio.mean().item()
            r_std = ratio.std().item()
            r_min = ratio.min().item()
            r_max = ratio.max().item()
            d_mean = default_vals.float().mean().item()

            print(
                f"  {desc:<25s} {d_mean:>12.4f} {r_mean:>12.4f}x "
                f"{r_std:>11.4f} {r_min:>10.4f} {r_max:>10.4f}"
            )
        except Exception as e:
            print(f"  {desc:<25s} [verify failed: {e}]")

    print()


def dr_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Update domain randomization ranges based on training progress.

    Each stage dict must have:
        - "step": int — common_step_counter threshold
        - "event_name": str — name of the EventTermCfg to modify
        - "ranges": tuple or dict — new ranges for dr.* functions

    Optionally:
        - "kp_range": tuple — for randomize_pd_gains events
        - "kd_range": tuple — for randomize_pd_gains events

    Args:
        env: The RL environment.
        env_ids: Unused (curriculum is global).
        stages: List of stage dicts, sorted by "step" ascending.

    Returns:
        Loggable dict with current DR range values.
    """
    del env_ids  # Curriculum is global.
    step = env.common_step_counter
    log = {}

    # Track stage transitions for verification logging
    if not hasattr(env, "_dr_curriculum_last_stage_step"):
        env._dr_curriculum_last_stage_step = -1

    for stage in stages:
        if step >= stage["step"]:
            event_name = stage["event_name"]
            term_cfg = env.event_manager.get_term_cfg(event_name)

            # Update ranges for dr.* events
            if "ranges" in stage:
                term_cfg.params["ranges"] = stage["ranges"]

            # Update ranges for randomize_pd_gains events
            if "kp_range" in stage:
                term_cfg.params["kp_range"] = stage["kp_range"]
            if "kd_range" in stage:
                term_cfg.params["kd_range"] = stage["kd_range"]

            # Update ranges for pseudo_inertia events
            for key in ("alpha_range", "d_range", "t_range"):
                if key in stage:
                    term_cfg.params[key] = stage[key]

    # Verify actual model parameters at stage transitions.
    _verify_dr_active(env, step, stages)

    # Log current ranges for training diagnostics (deduplicate by event_name).
    logged_events = set()
    for stage in stages:
        event_name = stage["event_name"]
        if event_name in logged_events:
            continue
        logged_events.add(event_name)
        term_cfg = env.event_manager.get_term_cfg(event_name)
        if "ranges" in term_cfg.params:
            r = term_cfg.params["ranges"]
            if isinstance(r, (tuple, list)):
                log[f"dr/{event_name}_min"] = torch.tensor(r[0])
                log[f"dr/{event_name}_max"] = torch.tensor(r[1])
        if "kp_range" in term_cfg.params:
            r = term_cfg.params["kp_range"]
            log[f"dr/{event_name}_kp_min"] = torch.tensor(r[0])
            log[f"dr/{event_name}_kp_max"] = torch.tensor(r[1])
        if "kd_range" in term_cfg.params:
            r = term_cfg.params["kd_range"]
            log[f"dr/{event_name}_kd_min"] = torch.tensor(r[0])
            log[f"dr/{event_name}_kd_max"] = torch.tensor(r[1])
        if "alpha_range" in term_cfg.params:
            r = term_cfg.params["alpha_range"]
            log[f"dr/{event_name}_alpha_min"] = torch.tensor(r[0])
            log[f"dr/{event_name}_alpha_max"] = torch.tensor(r[1])

    return log
