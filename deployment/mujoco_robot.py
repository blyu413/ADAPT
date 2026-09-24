"""MuJoCo state and position-control adapter for the 29-DoF G1."""

import mujoco
import numpy as np

from common.paths import ASSET_DIR
from deployment.config_manager import configure_model
from deployment.robot_interface import RobotInterface, RobotState


class MujocoRobot(RobotInterface):
    def __init__(self, cfg, *, seed=0, encoder_noise=False):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
        self.actuators = configure_model(self.model, cfg)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.encoder_noise = encoder_noise
        self.lin_adr = self.model.sensor("robot/imu_lin_vel").adr[0]
        self.ang_adr = self.model.sensor("robot/imu_ang_vel").adr[0]
        self.body_id = self.model.body("pelvis").id
        self.state = None

    def configure(self, cfg):
        """Apply the active policy's PD gains and timing without resetting the pose."""
        self.actuators = configure_model(self.model, cfg)
        self.cfg = cfg
        self.data.ctrl[self.actuators] = self.data.qpos[7:]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, default_angles=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.76]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        angles = self.cfg.default_position if default_angles is None else default_angles
        self.data.qpos[7:] = angles
        self.data.ctrl[self.actuators] = angles
        mujoco.mj_forward(self.model, self.data)
        self.bias = self.rng.uniform(-0.015, 0.015, 29) if self.encoder_noise else np.zeros(29)
        return self.read_state()

    def read_state(self):
        """Sample one control-rate state, including optional encoder noise."""
        q, v = self.data.qpos.copy(), self.data.qvel.copy()
        if self.encoder_noise:
            q[7:] += self.bias + self.rng.uniform(-1e-4, 1e-4, 29)
            v[6:] += self.rng.uniform(-0.03, 0.03, 29)
        self.state = RobotState(
            q,
            v,
            self.data.sensordata[self.lin_adr : self.lin_adr + 3].copy(),
            self.data.sensordata[self.ang_adr : self.ang_adr + 3].copy(),
            self.data.xmat[self.body_id].reshape(3, 3).copy(),
        )
        return self.state

    def get_state(self):
        return self.state

    def set_targets(self, targets):
        """Write joint-order targets into MuJoCo actuator order."""
        self.data.ctrl[self.actuators] = targets

    def advance(self):
        """Run the physics steps between two policy decisions."""
        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.model, self.data)

    def step(self, targets):
        """Apply one policy target, advance physics, and refresh robot state."""
        self.set_targets(targets)
        self.advance()
        return self.read_state()
