"""CPU ONNX policy inference with the matching deployment configuration."""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from deployment.config_manager import DeploymentConfig
from deployment.obs_processor import ObservationHistory


class Policy:
    def __init__(self, path):
        path = Path(path)
        self.cfg = DeploymentConfig.load(path.with_suffix(".json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.cfg.model_sha256:
            raise ValueError("ONNX and deployment JSON do not match; re-export both together")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, self.cfg.observation_dim]:
            raise ValueError("ONNX input shape does not match deployment configuration")
        if len(outputs) != 1 or outputs[0].shape != [1, len(self.cfg.joint_names)]:
            raise ValueError("ONNX output shape does not match G1 action dimension")
        self.input_name = inputs[0].name
        self.history = ObservationHistory(self.cfg)

    def step(self, state, command, residual):
        obs = self.history.compute(state, command, residual)
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite policy observation")
        action = self.session.run(None, {self.input_name: obs})[0][0]
        if not np.isfinite(action).all():
            raise ValueError("Non-finite policy action")
        self.history.last_action[:] = action
        return action * np.asarray(self.cfg.action_scale, np.float32) + np.asarray(
            self.cfg.default_position, np.float32
        )
