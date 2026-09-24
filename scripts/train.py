"""Train an ADAPT task with MJLab's single- or multi-GPU launcher."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("MUJOCO_GL", "egl")

import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

from tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from common.training import make_runner, restore


@dataclass
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    log_root: Path = Path("logs/rsl_rl")
    resume_checkpoint: Path | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def run_train(task, cfg, log_dir):
    """Worker entry point; torchrunx invokes one copy per selected GPU."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        rank = 0
        seed = cfg.agent.seed
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    if cfg.resume_checkpoint is not None and cfg.wandb_run_path is not None:
        raise ValueError("Use either --resume-checkpoint or --wandb-run-path, not both")

    configure_torch_backends()
    cfg.agent.seed = seed
    cfg.env.seed = seed
    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    checkpoint = cfg.resume_checkpoint
    if cfg.wandb_run_path is not None:
        checkpoint, was_cached = get_wandb_checkpoint_path(
            log_dir.parent,
            Path(cfg.wandb_run_path),
            cfg.wandb_checkpoint_name,
        )
        if rank == 0:
            source = "cached" if was_cached else "downloaded"
            print(f"[INFO] W&B checkpoint {source}: {checkpoint}")
    if cfg.agent.resume and checkpoint is None:
        raise ValueError(
            "This task requires --resume-checkpoint or --wandb-run-path "
            "(the matching Stage 1 checkpoint)"
        )

    env, runner = make_runner(
        task,
        cfg.env,
        cfg.agent,
        device,
        str(log_dir),
        rank=rank,
    )
    try:
        if checkpoint:
            restore(runner, env, checkpoint, device)
        if rank == 0:
            dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
            dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
            (log_dir / "task.txt").write_text(task + "\n")
        add_wandb_tags(cfg.agent.wandb_tags)
        runner.add_git_repo_to_log(__file__)
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
    if rank == 0:
        print(f"Saved training run: {log_dir}")


def launch_training(task, cfg):
    """Create one run directory and launch MJLab/RSL-RL workers."""
    suffix = "_" + cfg.agent.run_name if cfg.agent.run_name else ""
    log_dir = (
        cfg.log_root
        / cfg.agent.experiment_name
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + suffix)
    )
    log_dir.mkdir(parents=True, exist_ok=False)

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task, cfg, log_dir)
    else:
        import torchrunx

        logging.basicConfig(level=logging.INFO)
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            os.environ["TORCHRUNX_LOG_DIR"] = (
                cfg.torchrunx_log_dir
                if cfg.torchrunx_log_dir is not None
                else str(log_dir / "torchrunx")
            )
        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task, cfg, log_dir)
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
    launch_training(task, cfg)


if __name__ == "__main__":
    main()
