"""Single-environment first-order momentum observer for ADAPT.

The caller synchronizes qpos, qvel, and actuator-order position targets at
the control rate. This model computes nominal dynamics, not a future rollout.
The momentum-bias term uses the approximation beta = -qfrc_bias.
"""

import mujoco
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class MomentumObserver:
    def __init__(
        self, model, dt=0.02, gain=3.0, reference_body="pelvis", filter_order=2, cutoff_hz=1.0
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = float(dt)
        self.gain = np.broadcast_to(np.asarray(gain, dtype=np.float64), (model.nv,)).copy()
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, reference_body)
        if self.body_id < 0:
            raise ValueError(f"Unknown reference body: {reference_body}")
        self.nominal_mg = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        self.sos = (
            butter(filter_order, cutoff_hz, fs=1.0 / self.dt, output="sos")
            if filter_order
            else None
        )
        self.raw = np.zeros(model.nv)
        self.filtered = np.zeros(model.nv)
        self.previous_momentum = np.zeros(model.nv)
        self.initialized = False

    def reset(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_mulM(self.model, self.data, self.previous_momentum, self.data.qvel)
        self.raw[:] = 0.0
        self.raw[2] = self.nominal_mg
        self.filtered[:] = self.raw
        if self.sos is not None:
            self.filter_state = np.zeros((len(self.sos), 2, self.model.nv))
            self.filter_state[:, :, 2] = sosfilt_zi(self.sos) * self.nominal_mg
        self.initialized = True

    def update(self, qpos, qvel, ctrl):
        if not self.initialized:
            self.reset(qpos, qvel)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        momentum = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, momentum, self.data.qvel)
        known = self.data.qfrc_actuator + self.data.qfrc_passive - self.data.qfrc_bias
        self.raw = (
            self.raw
            + self.gain * (momentum - self.previous_momentum)
            - self.gain * self.dt * (known + self.raw)
        )
        self.previous_momentum[:] = momentum
        if self.sos is None:
            self.filtered[:] = self.raw
        else:
            filtered, self.filter_state = sosfilt(
                self.sos,
                self.raw.reshape(1, -1),
                axis=0,
                zi=self.filter_state,
            )
            self.filtered[:] = filtered[0]
        return self.filtered.copy()

    def local_residual(self, body_rotation=None):
        """Subtract nominal weight and apply the original base-frame convention.

        Output order: base force (N), base torque (Nm), joint residuals (Nm).
        The training implementation rotates both base triples by root rotation.
        """
        residual = self.filtered.copy()
        residual[2] -= self.nominal_mg
        rotation = (
            self.data.xmat[self.body_id].reshape(3, 3) if body_rotation is None else body_rotation
        ).T
        residual[:3] = rotation @ residual[:3]
        residual[3:6] = rotation @ residual[3:6]
        return residual.astype(np.float32)
