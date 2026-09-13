"""Single-GPU training with the original environment and PPO configuration overrides."""

from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from common.training import make_runner, restore


@dataclass
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    device: str = "cuda:0"
    log_root: Path = Path("logs/rsl_rl")
    resume_checkpoint: Path | None = None


def run_train(task, cfg):
    if cfg.agent.resume and cfg.resume_checkpoint is None:
        raise ValueError("This task requires --resume-checkpoint (the matching Stage 1 .pt)")
    configure_torch_backends()
    cfg.env.seed = cfg.agent.seed
    log_dir = (
        cfg.log_root
        / cfg.agent.experiment_name
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + cfg.agent.run_name)
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    env, runner = make_runner(task, cfg.env, cfg.agent, cfg.device, str(log_dir))
    try:
        if cfg.resume_checkpoint:
            restore(runner, env, cfg.resume_checkpoint, cfg.device)
        dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
        (log_dir / "task.txt").write_text(task + "\n")
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
    print(f"Saved training run: {log_dir}")
    return log_dir


def main():
    task, rest = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(
        TrainConfig,
        args=rest,
        default=TrainConfig(load_env_cfg(task), load_rl_cfg(task)),
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )
    run_train(task, cfg)


if __name__ == "__main__":
    main()
