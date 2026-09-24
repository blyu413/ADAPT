"""CPU ONNX policy inference with the matching deployment configuration."""

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from deployment.config_manager import DeploymentConfig, configure_model
from deployment.obs_processor import (
    MoMoObservationGenerator,
    ObservationGenerator,
    PolicyProcessor,
)
from observer import MomentumObserver


class BasePolicy(ABC):
    """One policy's deployment config, observation state, and action processor."""

    def __init__(self, cfg, name):
        self.name = name
        self.cfg = cfg
        self.obs_gen = self.make_observation_generator(cfg)
        self.policy_proc = PolicyProcessor(cfg)
        self.history = self.obs_gen  # Existing telemetry and real-robot entry point.
        self.observer = None

    def make_observation_generator(self, cfg):
        return ObservationGenerator(cfg)

    def attach_observer(self, model):
        """Use a separate nominal MuJoCo model for this policy's MoMo estimate."""
        self.actuators = configure_model(model, self.cfg)
        self.observer = MomentumObserver(
            model,
            dt=self.cfg.control_dt,
            gain=self.cfg.observer_gain,
            filter_order=self.cfg.filter_order,
            cutoff_hz=self.cfg.cutoff_hz,
        )
        return self.observer

    def reset(self, state=None):
        self.obs_gen.reset()
        if state is not None and self.observer is not None:
            self.observer.reset(state.qpos, state.qvel)
            self.refresh_encoder(state)

    def refresh_encoder(self, state):
        self.obs_gen.refresh_encoder(state)

    def update_momo(self, state, control_targets):
        """Post-control-rate observer update, also used for telemetry-only policies."""
        if self.observer is None:
            return
        ctrl = np.zeros(self.observer.model.nu)
        ctrl[self.actuators] = control_targets
        self.observer.update(state.qpos, state.qvel, ctrl)

    def apply_config_to_robot(self, robot):
        robot.configure(self.cfg)

    def on_activate(self):
        pass

    def on_deactivate(self):
        pass

    def local_residual(self, state):
        return None

    def build_observation(self, state, command):
        observation = self.obs_gen.get_obs(state, command, self.local_residual(state))
        if not np.isfinite(observation).all():
            raise ValueError("Non-finite policy observation")
        return observation

    @abstractmethod
    def infer(self, observation):
        """Return one raw actor action in joint order."""

    def action_to_targets(self, action):
        return self.policy_proc.process_action(action)

    def step(self, state, command):
        """Observation -> inference -> joint-order position targets."""
        observation = self.build_observation(state, command)
        action = self.infer(observation)
        self.obs_gen.update_last_action(action)
        return self.action_to_targets(action)


class VelocityPolicy(BasePolicy):
    """ONNX policy paired with the JSON snapshot exported from its task."""

    def __init__(self, path, name=None):
        path = Path(path)
        cfg = DeploymentConfig.load(path.with_suffix(".json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg.model_sha256:
            raise ValueError("ONNX and deployment JSON do not match; re-export both together")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, cfg.observation_dim]:
            raise ValueError("ONNX input shape does not match deployment configuration")
        if len(outputs) != 1 or outputs[0].shape != [1, len(cfg.joint_names)]:
            raise ValueError("ONNX output shape does not match G1 action dimension")
        self.input_name = inputs[0].name
        super().__init__(cfg, name or path.stem)

    def infer(self, observation):
        """Run the ONNX actor, including its exported observation normalizer."""
        action = self.session.run(None, {self.input_name: observation})[0][0]
        if not np.isfinite(action).all():
            raise ValueError("Non-finite policy action")
        return action


class VelocityMoMoPolicy(VelocityPolicy):
    """ONNX velocity policy with the exported ADAPT MoMo observation term."""

    def make_observation_generator(self, cfg):
        return MoMoObservationGenerator(cfg)

    def local_residual(self, state):
        if self.observer is None:
            raise RuntimeError("MoMo policy needs an attached nominal model")
        return self.observer.local_residual(state.body_rotation)


class ZeroPolicy(BasePolicy):
    """Default-position fallback sharing the same config and switching interface."""

    def __init__(self, cfg, name="Zero"):
        super().__init__(cfg, name)

    def make_observation_generator(self, cfg):
        return MoMoObservationGenerator(cfg) if cfg.residual_mode != "none" else ObservationGenerator(cfg)

    def local_residual(self, state):
        if self.cfg.residual_mode == "full":
            if self.observer is None:
                raise RuntimeError("MoMo observation needs an attached nominal model")
            return self.observer.local_residual(state.body_rotation)
        return None

    def infer(self, observation):
        return np.zeros(len(self.cfg.joint_names), dtype=np.float32)


class Policy(VelocityMoMoPolicy):
    """Compatibility entry point; new callers should use load_policy()."""


def load_policy(path, name=None):
    """Select the policy implementation from the ONNX-matched JSON snapshot."""
    cfg = DeploymentConfig.load(Path(path).with_suffix(".json"))
    cls = VelocityMoMoPolicy if cfg.residual_mode != "none" else VelocityPolicy
    return cls(path, name=name)


class PolicyManager:
    """The old register/activate/next/prev interface, using ONNX/JSON policies."""

    def __init__(self):
        self._policies = {}
        self._order = []
        self._active_name = None

    def register(self, policy, set_active=False):
        if policy.name in self._policies:
            raise ValueError(f"Policy already registered: {policy.name}")
        self._policies[policy.name] = policy
        self._order.append(policy.name)
        if set_active:
            self.set_active(policy.name)

    @property
    def active_policy(self):
        if self._active_name is None:
            raise RuntimeError("No active policy")
        return self._policies[self._active_name]

    @property
    def active_policy_name(self):
        return self._active_name

    def list_policies(self):
        return tuple(self._order)

    def set_active(self, name, robot=None):
        if name not in self._policies:
            raise ValueError(f"Unknown policy: {name}")
        policy = self._policies[name]
        if robot is not None:
            policy.apply_config_to_robot(robot)
        if self._active_name is not None:
            self.active_policy.on_deactivate()
        self._active_name = name
        policy.on_activate()
        policy.reset(None if robot is None else robot.state)
        return name

    def next(self, robot=None):
        index = self._order.index(self._active_name)
        return self.set_active(self._order[(index + 1) % len(self._order)], robot)

    def prev(self, robot=None):
        index = self._order.index(self._active_name)
        return self.set_active(self._order[(index - 1) % len(self._order)], robot)

    def step(self, state, command):
        return self.active_policy.step(state, command)

    def refresh_encoder(self, state):
        self.active_policy.refresh_encoder(state)
