"""Versioned JSONL/UDP records; force in N, torque in Nm, velocity in m/s."""

import json
from pathlib import Path
import socket
import time

import numpy as np


class Telemetry:
    def __init__(self, log=None, udp=None):
        self.file = None
        if log:
            path = Path(log)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("x")
        self.address = None
        self.socket = None
        if udp:
            host, port = udp.rsplit(":", 1)
            self.address = (host, int(port))
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def write(self, record):
        payload = json.dumps({"schema": 1, "wall_time": time.time(), **record}, allow_nan=False)
        if self.file:
            self.file.write(payload + "\n")
            self.file.flush()
        if self.socket:
            self.socket.sendto(payload.encode(), self.address)

    def close(self):
        if self.file:
            self.file.close()
        if self.socket:
            self.socket.close()


def observer_record(observer, policy, state, command, local, **extra):
    scaled = local / np.asarray(policy.cfg.residual_scale, np.float32)
    leg = scaled[6:18].reshape(2, 6)
    return dict(
        task=policy.cfg.task,
        raw_generalized=observer.raw.tolist(),
        filtered_generalized=observer.filtered.tolist(),
        local_residual=local.tolist(),
        scaled_residual=scaled.tolist(),
        leg_envelope=np.sqrt(np.mean(leg**2, axis=1)).tolist(),
        actor_residual=policy.history.residual_term(local).tolist(),
        linear_velocity=state.linear_velocity.tolist(),
        angular_velocity=state.angular_velocity.tolist(),
        command=np.asarray(command).tolist(),
        **extra,
    )
