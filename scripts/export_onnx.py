"""Export a trusted .pt checkpoint and its matching lightweight deployment JSON."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from deployment.config_manager import DeploymentConfig
from common.paths import ASSET_DIR
from tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from common.training import make_runner, restore


def export(task, checkpoint, output, device):
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env, runner = make_runner(task, cfg, load_rl_cfg(task), device)
    try:
        restore(runner, env, checkpoint, device)
        runner.export_policy_to_onnx(str(output.parent), output.name)
        attach_metadata_to_onnx(str(output), get_base_metadata(env.unwrapped, run_path="exported"))
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
        deployment = DeploymentConfig.from_env(task, cfg, model)
        deployment.save(output.with_suffix(".json"), output)
        # Verify the complete normalizer + actor export against real task observations.
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        policy = runner.get_inference_policy(device=device)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(5):
                actions = policy(obs)
                actual = session.run(
                    None, {session.get_inputs()[0].name: obs["actor"].cpu().numpy()}
                )[0]
                np.testing.assert_allclose(actual, actions.cpu().numpy(), atol=2e-5, rtol=2e-5)
                obs, _, _, _ = env.step(actions)
        print(f"Export validated: {output} + {output.with_suffix('.json')}")
    finally:
        env.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list_tasks())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export(args.task, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
