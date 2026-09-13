"""Train stage 1 then resume its last checkpoint into the matching hand-load task."""

import argparse
from pathlib import Path

from tasks.registry import load_env_cfg, load_rl_cfg
from scripts.train import TrainConfig, run_train

PAIRS = {
    "adapt": (
        "MOMO-Lumped-Flat-Unitree-G1-EffortScale",
        "MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale",
    ),
    "baseline": ("Ref-NoMoMo-Flat-Unitree-G1", "Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=PAIRS)
    parser.add_argument(
        "--stage1-checkpoint", type=Path, help="Skip stage 1 and continue this checkpoint"
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--stage1-iterations", type=int, default=15000)
    parser.add_argument("--stage2-iterations", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    args = parser.parse_args()
    checkpoint = args.stage1_checkpoint
    for stage, task in enumerate(PAIRS[args.policy], 1):
        if stage == 1 and checkpoint:
            continue
        cfg = TrainConfig(
            load_env_cfg(task), load_rl_cfg(task), args.device, args.log_root, checkpoint
        )
        cfg.env.scene.num_envs = args.num_envs
        cfg.agent.run_name = f"stage{stage}"
        cfg.agent.max_iterations = args.stage1_iterations if stage == 1 else args.stage2_iterations
        folder = run_train(task, cfg)
        checkpoint = max(folder.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
