"""Baseline, ADAPT, and light-step task registrations."""

from .env_cfg import (
    unitree_g1_momo_lumped_flat_ref_nomomo_env_cfg,
    unitree_g1_momo_lumped_flat_ref_nomomo_hand_load_clean_mix_env_cfg,
    unitree_g1_momo_lumped_flat_effort_scale_env_cfg,
    unitree_g1_momo_lumped_flat_hand_load_clean_mix_effort_scale_env_cfg,
    unitree_g1_momo_lumped_flat_leg_effort_softstep_no_actor_feedback_env_cfg,
)
from .rl_cfg import (
    momo_g1_ppo_ref_nomomo_runner_cfg,
    momo_g1_ppo_ref_nomomo_hand_load_clean_mix_runner_cfg,
    momo_g1_ppo_effort_scale_runner_cfg,
    momo_g1_ppo_hand_load_clean_mix_effort_scale_runner_cfg,
    momo_g1_ppo_leg_effort_softstep_no_actor_feedback_runner_cfg,
)
from .momo_env import MoMoEnv
from .effort_scale_runner import MoMoEffortScaleRunner
from ..registry import register_mjlab_task


class MoMoEnvStage1Noise(MoMoEnv):
    """Stage 1 encoder noise used by the released checkpoints."""

    _qpos_noise_range = 1e-4
    _qvel_noise_range = 0.03
    _encoder_bias_range = 0.015


TASKS = (
    (
        "Ref-NoMoMo-Flat-Unitree-G1",
        unitree_g1_momo_lumped_flat_ref_nomomo_env_cfg,
        momo_g1_ppo_ref_nomomo_runner_cfg,
        MoMoEnvStage1Noise,
        None,
    ),
    (
        "Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix",
        unitree_g1_momo_lumped_flat_ref_nomomo_hand_load_clean_mix_env_cfg,
        momo_g1_ppo_ref_nomomo_hand_load_clean_mix_runner_cfg,
        MoMoEnv,
        None,
    ),
    (
        "MOMO-Lumped-Flat-Unitree-G1-EffortScale",
        unitree_g1_momo_lumped_flat_effort_scale_env_cfg,
        momo_g1_ppo_effort_scale_runner_cfg,
        MoMoEnvStage1Noise,
        MoMoEffortScaleRunner,
    ),
    (
        "MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale",
        unitree_g1_momo_lumped_flat_hand_load_clean_mix_effort_scale_env_cfg,
        momo_g1_ppo_hand_load_clean_mix_effort_scale_runner_cfg,
        MoMoEnv,
        MoMoEffortScaleRunner,
    ),
    (
        "MOMO-Lumped-Flat-Unitree-G1-LegEffortSoftStep-NoActorFeedback",
        unitree_g1_momo_lumped_flat_leg_effort_softstep_no_actor_feedback_env_cfg,
        momo_g1_ppo_leg_effort_softstep_no_actor_feedback_runner_cfg,
        MoMoEnvStage1Noise,
        MoMoEffortScaleRunner,
    ),
)

for task_id, make_env, make_rl, env_cls, runner_cls in TASKS:
    rl_cfg = make_rl()
    rl_cfg.logger = "tensorboard"
    rl_cfg.upload_model = False
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_env(),
        play_env_cfg=make_env(play=True),
        rl_cfg=rl_cfg,
        env_cls=env_cls,
        runner_cls=runner_cls,
    )
