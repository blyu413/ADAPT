from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def momo_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Shared PPO settings for the five released tasks."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="momo_g1_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=15_000,
    )


def momo_g1_ppo_effort_scale_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 1 EffortScale walker: train from scratch."""
    cfg = momo_g1_ppo_runner_cfg()
    cfg.resume = False
    cfg.experiment_name = "momo_g1_effort_scale"
    cfg.run_name = "stage1"
    return cfg


def momo_g1_ppo_leg_effort_softstep_no_actor_feedback_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """LegEffortSoftStep control: actor sees zero MoMo feedback."""
    cfg = momo_g1_ppo_effort_scale_runner_cfg()
    cfg.experiment_name = "momo_g1_leg_effort_softstep_no_actor_feedback"
    return cfg


def momo_g1_ppo_hand_load_clean_mix_effort_scale_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 2 EffortScale HandLoad-CleanMix.

    Pass the matching Stage 1 checkpoint to train.py with --resume-checkpoint.
    """
    cfg = momo_g1_ppo_runner_cfg()
    cfg.resume = True
    cfg.load_run = ".*stage1$"
    cfg.load_checkpoint = "model_.*.pt"
    cfg.experiment_name = "momo_g1_effort_scale"
    cfg.run_name = "hand_load_clean_mix_stage2"
    cfg.max_iterations = 20_000
    return cfg


def momo_g1_ppo_ref_nomomo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 1 proprioception-only baseline."""
    cfg = momo_g1_ppo_runner_cfg()
    cfg.experiment_name = "momo_g1_velocity_ref_nomomo"
    return cfg


def momo_g1_ppo_ref_nomomo_hand_load_clean_mix_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 2 proprioception-only baseline with HandLoad-CleanMix.

    Pass the matching Stage 1 checkpoint to train.py with --resume-checkpoint.
    """
    cfg = momo_g1_ppo_runner_cfg()
    cfg.resume = True
    cfg.load_run = ".*"
    cfg.load_checkpoint = "model_.*.pt"
    cfg.experiment_name = "momo_g1_hand_load_ref_nomomo_clean_mix"
    cfg.max_iterations = 20_000
    return cfg
