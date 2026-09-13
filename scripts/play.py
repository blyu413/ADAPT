"""Inspect a trusted .pt policy in its training environment."""

import argparse
import torch

from tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from common.training import make_runner, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list_tasks())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--viewer", choices=["native", "viser", "none"], default="native")
    parser.add_argument("--steps", type=int, default=100, help="Headless step count")
    args = parser.parse_args()
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    env, runner = make_runner(args.task, cfg, load_rl_cfg(args.task), args.device)
    try:
        restore(runner, env, args.checkpoint, args.device)
        policy = runner.get_inference_policy(device=args.device)
        if args.viewer == "none":
            obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, reward, _, _ = env.step(policy(obs))
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("Non-finite rollout reward")
            print(f"Completed {args.steps} policy steps")
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

            viewer_cls = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
            viewer_cls(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
