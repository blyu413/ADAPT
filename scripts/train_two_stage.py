"""Train stage 1 then resume its last checkpoint into the matching hand-load task."""

import argparse
import json
from pathlib import Path

from tasks.registry import load_env_cfg, load_rl_cfg
from scripts.train import TrainConfig, launch_training

PAIRS = {
    "adapt": (
        "MOMO-Lumped-Flat-Unitree-G1-EffortScale",
        "MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale",
    ),
    "baseline": ("Ref-NoMoMo-Flat-Unitree-G1", "Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix"),
}


def gpu_selection(value):
    """Parse MJLab-style --gpu-ids values: '[0, 1]', 'all', or 'none'."""
    if value == "all":
        return "all"
    if value == "none":
        return None
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("Use '[0, 1]', 'all', or 'none'") from exc
    if not isinstance(result, list) or not all(isinstance(item, int) for item in result):
        raise argparse.ArgumentTypeError("--gpu-ids must be a JSON list of integers")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=PAIRS)
    parser.add_argument(
        "--stage1-checkpoint", type=Path, help="Skip stage 1 and continue this checkpoint"
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--stage1-iterations", type=int, default=15000)
    parser.add_argument("--stage2-iterations", type=int, default=20000)
    parser.add_argument(
        "--gpu-ids",
        type=gpu_selection,
        default=[0],
        help="MJLab GPU selection: '[0]', '[0, 1]', 'all', or 'none' for CPU",
    )
    parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--wandb-project", default="adapt")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--upload-model", action="store_true")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    args = parser.parse_args()
    checkpoint = args.stage1_checkpoint
    for stage, task in enumerate(PAIRS[args.policy], 1):
        if stage == 1 and checkpoint:
            continue
        cfg = TrainConfig(
            env=load_env_cfg(task),
            agent=load_rl_cfg(task),
            log_root=args.log_root,
            resume_checkpoint=checkpoint,
            gpu_ids=args.gpu_ids,
        )
        cfg.env.scene.num_envs = args.num_envs
        cfg.agent.run_name = f"stage{stage}"
        cfg.agent.max_iterations = args.stage1_iterations if stage == 1 else args.stage2_iterations
        cfg.agent.logger = args.logger
        cfg.agent.wandb_project = args.wandb_project
        cfg.agent.wandb_tags = tuple(args.wandb_tags)
        cfg.agent.upload_model = args.upload_model
        folder = launch_training(task, cfg)
        checkpoint = max(folder.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
