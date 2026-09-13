"""Run the observer on a fixed-PD G1 trajectory; no policy or training import."""

import mujoco
import numpy as np

from deployment.config_manager import DeploymentConfig, configure_model
from observer import MomentumObserver
from common.paths import ASSET_DIR, checkpoint_path


def main():
    cfg = DeploymentConfig.load(checkpoint_path("adapt", ".json"))
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
    actuators = configure_model(model, cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[[0, 0, 0.76, 1, 0, 0, 0], cfg.default_position]
    data.ctrl[actuators] = cfg.default_position
    mujoco.mj_forward(model, data)
    observer = MomentumObserver(model, dt=cfg.control_dt)
    observer.reset(data.qpos, data.qvel)
    for step in range(250):
        for _ in range(cfg.decimation):
            mujoco.mj_step(model, data)
        observer.update(data.qpos, data.qvel, data.ctrl)
        if step % 50 == 0:
            print(f"t={data.time:.2f}s base residual [N]: {observer.local_residual()[:3].round(2)}")


if __name__ == "__main__":
    main()
