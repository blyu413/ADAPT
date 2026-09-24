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


def _base_lin_vel(generator, state, command, residual):
    return state.linear_velocity


def _base_ang_vel(generator, state, command, residual):
    return state.angular_velocity


def _projected_gravity(generator, state, command, residual):
    return inverse_rotate(state.qpos[3:7], [0, 0, -1])


def _joint_pos(generator, state, command, residual):
    return state.qpos[7:].astype(np.float32) - np.asarray(
        generator.cfg.default_position, np.float32
    )


def _joint_vel(generator, state, command, residual):
    return state.qvel[6:]


def _actions(generator, state, command, residual):
    return generator.last_action


def _command(generator, state, command, residual):
    return command


def _momo_disturbance(generator, state, command, residual):
    return generator.residual_term(residual)


OBS_HANDLERS = {
    "base_lin_vel": _base_lin_vel,
    "base_ang_vel": _base_ang_vel,
    "projected_gravity": _projected_gravity,
    "joint_pos": _joint_pos,
    "joint_vel": _joint_vel,
    "actions": _actions,
    "command": _command,
    "momo_disturbance": _momo_disturbance,
}


class ObservationGenerator:
    """Generate actor observations and maintain each term's history."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.buffers = {}
        self.last_action = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        self.encoder_state = None

    def refresh_encoder(self, state):
        """Share the already sampled robot state with observation and MoMo."""
        self.encoder_state = state

    def residual_term(self, local):
        return np.empty(0, dtype=np.float32)

    def compute(self, state, command, local_residual=None):
        result = []
        for spec in self.cfg.observations:
            name = spec["name"]
            value = np.asarray(
                OBS_HANDLERS[name](self, state, command, local_residual), dtype=np.float32
            ) * np.asarray(spec["scale"], np.float32)
            if value.shape != (spec["dim"],):
                raise ValueError(f"{name}: expected {spec['dim']} values, got {value.shape}")
            if name not in self.buffers:
                self.buffers[name] = np.tile(value, (spec["history"], 1))
            else:
                self.buffers[name][:-1] = self.buffers[name][1:]
                self.buffers[name][-1] = value
            result.append(self.buffers[name].ravel())
        return np.concatenate(result)[None, :]

    def get_obs(self, state, command, local_residual=None):
        if self.encoder_state is not state:
            self.refresh_encoder(state)
        return self.compute(self.encoder_state, command, local_residual)

    def update_last_action(self, action):
        self.last_action[:] = action


class MoMoObservationGenerator(ObservationGenerator):
    """ADAPT disturbance term; estimator updates remain outside get_obs()."""

    def residual_term(self, local):
        if self.cfg.residual_mode == "full":
            return np.asarray(local, np.float32) / np.asarray(
                self.cfg.residual_scale, np.float32
            )
        if self.cfg.residual_mode == "zero_leg":
            return np.zeros(14, dtype=np.float32)
        return super().residual_term(local)


class PolicyProcessor:
    """Map raw actor actions to joint-order position targets."""

    def __init__(self, cfg):
        self.action_scale = np.asarray(cfg.action_scale, np.float32)
        self.default_position = np.asarray(cfg.default_position, np.float32)

    def process_action(self, action):
        return action * self.action_scale + self.default_position


# Preserve the original ADAPT import for downstream users.
ObservationHistory = ObservationGenerator
