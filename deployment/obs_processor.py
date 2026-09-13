"""Term-major observation history shared by MuJoCo and G1."""

import numpy as np


def inverse_rotate(q, v):
    w, x, y, z = np.asarray(q, dtype=np.float32)
    return np.array(
        [
            v[0] * (w * w + x * x - y * y - z * z)
            + v[1] * 2 * (x * y + w * z)
            + v[2] * 2 * (x * z - w * y),
            v[0] * 2 * (x * y - w * z)
            + v[1] * (w * w - x * x + y * y - z * z)
            + v[2] * 2 * (y * z + w * x),
            v[0] * 2 * (x * z + w * y)
            + v[1] * 2 * (y * z - w * x)
            + v[2] * (w * w - x * x - y * y + z * z),
        ],
        dtype=np.float32,
    )


class ObservationHistory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.buffers = {}
        self.last_action = np.zeros(len(self.cfg.joint_names), dtype=np.float32)

    def residual_term(self, local):
        if self.cfg.residual_mode == "full":
            return np.asarray(local, np.float32) / np.asarray(self.cfg.residual_scale, np.float32)
        if self.cfg.residual_mode == "zero_leg":
            return np.zeros(14, dtype=np.float32)
        return np.empty(0, dtype=np.float32)

    def compute(self, state, command, local_residual):
        terms = dict(
            base_lin_vel=state.linear_velocity,
            base_ang_vel=state.angular_velocity,
            projected_gravity=inverse_rotate(state.qpos[3:7], [0, 0, -1]),
            joint_pos=state.qpos[7:].astype(np.float32)
            - np.asarray(self.cfg.default_position, np.float32),
            joint_vel=state.qvel[6:],
            actions=self.last_action,
            command=command,
            momo_disturbance=self.residual_term(local_residual),
        )
        result = []
        for spec in self.cfg.observations:
            name = spec["name"]
            value = np.asarray(terms[name], dtype=np.float32) * np.asarray(
                spec["scale"], np.float32
            )
            if value.shape != (spec["dim"],):
                raise ValueError(f"{name}: expected {spec['dim']} values, got {value.shape}")
            if name not in self.buffers:
                self.buffers[name] = np.tile(value, (spec["history"], 1))
            else:
                self.buffers[name][:-1] = self.buffers[name][1:]
                self.buffers[name][-1] = value
            result.append(self.buffers[name].ravel())
        return np.concatenate(result)[None, :]
