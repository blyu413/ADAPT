"""Minimal flat-ground G1 deployment, independent of mjlab and the robot SDK."""

import argparse
from contextlib import nullcontext
import time

import mujoco
import numpy as np

from common.telemetry import Telemetry, observer_record
from observer import MomentumObserver
from common.paths import ASSET_DIR, checkpoint_path
from deployment.config_manager import configure_model
from deployment.policy_manager import Policy
from deployment.robot_interface import RobotState


class Simulation:
    def __init__(self, checkpoint, *, seed=0, encoder_noise=False):
        self.policy = Policy(checkpoint)
        self.cfg = self.policy.cfg
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
        self.actuators = configure_model(self.model, self.cfg)
        nominal = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
        configure_model(nominal, self.cfg)
        self.observer = MomentumObserver(
            nominal,
            dt=self.cfg.control_dt,
            gain=self.cfg.observer_gain,
            filter_order=self.cfg.filter_order,
            cutoff_hz=self.cfg.cutoff_hz,
        )
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.encoder_noise = encoder_noise
        self.lin_adr = self.model.sensor("robot/imu_lin_vel").adr[0]
        self.ang_adr = self.model.sensor("robot/imu_ang_vel").adr[0]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.76]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:] = self.cfg.default_position
        self.data.ctrl[self.actuators] = self.cfg.default_position
        mujoco.mj_forward(self.model, self.data)
        self.bias = self.rng.uniform(-0.015, 0.015, 29) if self.encoder_noise else np.zeros(29)
        self.policy.history.reset()
        self._refresh_state()
        self.observer.reset(self.state.qpos, self.state.qvel)

    def _refresh_state(self):
        q, v = self.data.qpos.copy(), self.data.qvel.copy()
        if self.encoder_noise:
            q[7:] += self.bias + self.rng.uniform(-1e-4, 1e-4, 29)
            v[6:] += self.rng.uniform(-0.03, 0.03, 29)
        self.state = RobotState(
            q,
            v,
            self.data.sensordata[self.lin_adr : self.lin_adr + 3].copy(),
            self.data.sensordata[self.ang_adr : self.ang_adr + 3].copy(),
        )

    def local_residual(self):
        return self.observer.local_residual(self.data.xmat[self.observer.body_id].reshape(3, 3))

    def step(self, command):
        targets = self.policy.step(self.state, command, self.local_residual())
        self.data.ctrl[self.actuators] = targets
        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.model, self.data)
        self._refresh_state()
        self.observer.update(self.state.qpos, self.state.qvel, self.data.ctrl)
        return observer_record(
            self.observer,
            self.policy,
            self.state,
            command,
            self.local_residual(),
            source="mujoco",
            time=float(self.data.time),
            root_height=float(self.data.qpos[2]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["adapt", "baseline", "lightstep"], default="adapt")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--command", nargs=3, type=float, default=[0.8, 0, 0], metavar=("VX", "VY", "WZ")
    )
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--encoder-noise", action="store_true", help="Shared Stage 1 encoder noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--udp", help="Telemetry host:port, e.g. 127.0.0.1:9870")
    args = parser.parse_args()
    sim = Simulation(
        args.checkpoint or checkpoint_path(args.policy),
        seed=args.seed,
        encoder_noise=args.encoder_noise,
    )
    telemetry = Telemetry(args.log, args.udp)
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer

        context = mujoco.viewer.launch_passive(sim.model, sim.data)
    steps = int(args.seconds / sim.cfg.control_dt)
    if steps < 1:
        raise ValueError("--seconds must cover at least one policy step")
    min_height = float("inf")
    try:
        with context as viewer:
            for _ in range(steps):
                start = time.monotonic()
                if viewer and not viewer.is_running():
                    break
                record = sim.step(np.asarray(args.command, np.float32))
                telemetry.write(record)
                min_height = min(min_height, record["root_height"])
                if not np.isfinite(sim.data.qpos).all():
                    raise RuntimeError("Simulation state became non-finite")
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, sim.cfg.control_dt - (time.monotonic() - start)))
    finally:
        telemetry.close()
    print(f"{sim.cfg.task}: simulated {sim.data.time:.2f}s, minimum root height {min_height:.3f}m")


if __name__ == "__main__":
    main()
