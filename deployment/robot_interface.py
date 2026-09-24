"""G1 DDS adapter. Construction is read-only; control must be enabled explicitly."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import time

import numpy as np

from common.command_helper import initialize_command, position_command, damping_command
from common.crc import command_crc
from deployment.obs_processor import inverse_rotate


class RobotInterface(ABC):
    """The deployment-facing state, control, and reset interface."""

    @abstractmethod
    def get_state(self):
        """Return the latest robot state."""

    @abstractmethod
    def step(self, control_input):
        """Apply one joint-order control target."""

    @abstractmethod
    def reset(self, default_angles=None):
        """Reset simulation state; physical robots cannot be reset this way."""


@dataclass
class RobotState:
    qpos: np.ndarray
    qvel: np.ndarray
    linear_velocity: np.ndarray  # pelvis frame
    angular_velocity: np.ndarray  # pelvis frame
    body_rotation: np.ndarray | None = None  # MuJoCo pelvis rotation, if available


G1_MOTOR_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def motor_mapping(names):
    return [G1_MOTOR_NAMES.index(name) for name in names]


def state_from_message(msg, mapping, velocity):
    q = np.zeros(36)
    v = np.zeros(35)
    q[:3] = [0, 0, 0.76]  # Absolute translation does not enter the momentum observer.
    q[3:7] = msg.imu_state.quaternion  # Unitree: wxyz
    if not np.isclose(np.linalg.norm(q[3:7]), 1, atol=0.01):
        raise ValueError("Invalid IMU quaternion")
    q[7:] = [msg.motor_state[i].q for i in mapping]
    v[6:] = [msg.motor_state[i].dq for i in mapping]
    v[3:6] = msg.imu_state.gyroscope
    conjugate = q[3:7] * [1, -1, -1, -1]
    v[:3] = inverse_rotate(conjugate, velocity)
    if not np.isfinite(np.concatenate([q, v])).all():
        raise ValueError("Non-finite robot state")
    return RobotState(q, v, np.asarray(velocity).copy(), v[3:6].copy())


class G1(RobotInterface):
    timeout = 0.1  # Local monotonic age of advancing state/LIO samples.

    def __init__(self, cfg, model, interface, lidar_frame, lio_topic):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
        from deployment.fastlio_receiver_ros2 import FastLIOReceiver
        from deployment.pelvis_vel_estimator import PelvisVelocityEstimator

        self.cfg = cfg
        self.model = model
        self.mapping = motor_mapping(cfg.joint_names)
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.message = self.command_message = None
        self.state_time = self.command_time = 0.0
        self.publisher = None
        self.fault = None
        self.last_send = 0.0
        self.stop = threading.Event()
        self.watchdog = None
        self.estimator = PelvisVelocityEstimator(
            model, lidar_mount_extra_rpy=np.array([np.pi if lidar_frame == "raw" else 0, 0, 0])
        )
        ChannelFactoryInitialize(0, interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._state_callback, 1)
        self.command_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.command_subscriber.Init(self._command_callback, 1)
        self.lio = FastLIOReceiver(topic_name=lio_topic)

    def _state_callback(self, msg):
        with self.lock:
            if self.message is None or msg.tick != self.message.tick:
                self.message = msg
                self.state_time = time.monotonic()

    def _command_callback(self, msg):
        with self.lock:
            self.command_message = msg
            self.command_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            msg, stamp = self.message, self.state_time
        lio = self.lio.get_latest()
        now = time.monotonic()
        if msg is None or now - stamp > self.timeout:
            raise TimeoutError("Missing/stale rt/lowstate")
        if lio is None or now - lio["received_monotonic"] > self.timeout:
            raise TimeoutError("Missing/stale advancing external LIO odometry")
        state = state_from_message(msg, self.mapping, np.zeros(3))
        velocity, _ = self.estimator.estimate(
            state.qpos[7:], state.qvel[6:], lio["v_local"], lio["omega_local"]
        )
        state = state_from_message(msg, self.mapping, velocity)
        return state, dict(
            lowstate_age=now - stamp,
            lio_age=now - lio["received_monotonic"],
            lio_timestamp=lio["timestamp"],
            mode_machine=int(msg.mode_machine),
        )

    def get_state(self):
        return self.snapshot()[0]

    def step(self, control_input):
        self.send(control_input)

    def reset(self, default_angles=None):
        raise RuntimeError("A physical G1 cannot be reset through RobotInterface")

    def observed_targets(self):
        """Read-only observer requires the same nominal position-PD command contract."""
        with self.lock:
            msg, stamp = self.command_message, self.command_time
        if msg is None or time.monotonic() - stamp > self.timeout:
            raise TimeoutError("Observer needs fresh rt/lowcmd; use --state-only to inspect state")
        motors = [msg.motor_cmd[i] for i in self.mapping]
        if (
            msg.mode_pr != 0
            or any(m.mode != 1 for m in motors)
            or not np.allclose([m.kp for m in motors], self.cfg.stiffness)
            or not np.allclose([m.kd for m in motors], self.cfg.damping)
            or np.any([m.dq != 0 or m.tau != 0 for m in motors])
        ):
            raise ValueError(
                "Observed lowcmd is not the released position-PD contract; use --state-only"
            )
        return np.asarray([m.q for m in motors])

    def enable_control(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        _, status = self.snapshot()
        # This asset describes the original 29-DoF G1, not the revised reduction ratio.
        if status["mode_machine"] != 2:
            raise ValueError(
                "This release's G1 asset requires mode_machine=2; check your hardware revision"
            )
        self.expected_mode = status["mode_machine"]
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        initialize_command(self.cmd, self.expected_mode)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.last_send = time.monotonic()
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _publish(self):
        self.cmd.crc = command_crc(self.cmd)
        if not self.publisher.Write(self.cmd):
            raise RuntimeError("DDS command write failed")

    def send(self, targets):
        if self.publisher is None:
            raise RuntimeError("Read-only adapter cannot send motor commands")
        targets = np.asarray(targets)
        if targets.shape != (29,) or not np.isfinite(targets).all():
            raise ValueError("Invalid joint target")
        # These are PD equilibrium targets, not measured joint angles. The trained
        # controller can place them outside geometric limits to request torque.
        with self.send_lock:
            if self.fault:
                raise RuntimeError(self.fault)
            self.snapshot()  # No control using stale sensor data.
            position_command(self.cmd, self.mapping, targets, self.cfg.stiffness, self.cfg.damping)
            self._publish()
            self.last_send = time.monotonic()

    def _watch(self):
        while not self.stop.wait(0.01):
            lio = self.lio.get_latest()
            now = time.monotonic()
            with self.lock:
                state_time, msg = self.state_time, self.message
            stale = (
                now - self.last_send > self.timeout
                or now - state_time > self.timeout
                or lio is None
                or now - lio["received_monotonic"] > self.timeout
                or msg.mode_machine != self.expected_mode
            )
            if stale:
                with self.send_lock:
                    self.fault = "Command/state/LIO watchdog or hardware mode changed"
                    damping_command(self.cmd, self.mapping)
                    self._publish()
                return

    def close(self):
        self.stop.set()
        if self.watchdog:
            self.watchdog.join(timeout=1)
        if self.publisher:
            with self.send_lock:
                damping_command(self.cmd, self.mapping)
                for _ in range(25):
                    self._publish()
                    time.sleep(self.cfg.control_dt)
            self.publisher.Close()
        self.subscriber.Close()
        self.command_subscriber.Close()
