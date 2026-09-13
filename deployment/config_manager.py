"""Portable deployment settings. Loading these does not import the training stack."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np


@dataclass
class DeploymentConfig:
    task: str
    joint_names: list[str]
    timestep: float
    decimation: int
    stiffness: list[float]
    damping: list[float]
    armature: list[float]
    effort_limits: list[float]
    default_position: list[float]
    action_scale: list[float]
    observations: list[dict]
    residual_mode: str
    residual_scale: list[float]
    observer_gain: float = 3.0
    filter_order: int = 2
    cutoff_hz: float = 1.0
    model_sha256: str = ""

    @property
    def control_dt(self):
        return self.timestep * self.decimation

    @property
    def observation_dim(self):
        return sum(term["dim"] * term["history"] for term in self.observations)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path, model_path):
        self.model_sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_env(cls, task, env, model):
        """Extract only the five released tasks' deployment contract."""
        names = [
            model.joint(i).name
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        short = [name.split("/")[-1] for name in names]
        robot = env.scene.entities["robot"]

        def gains(field):
            return np.asarray(
                [
                    next(
                        getattr(group, field)
                        for group in robot.articulation.actuators
                        if any(re.fullmatch(p, name) for p in group.target_names_expr)
                    )
                    for name in short
                ],
                dtype=np.float32,
            ).tolist()

        def patterns(values, default=0.0):
            if not isinstance(values, dict):
                return np.full(len(names), values, dtype=np.float32).tolist()
            return np.asarray(
                [
                    next((v for p, v in values.items() if re.fullmatch(p, name)), default)
                    for name in short
                ],
                dtype=np.float32,
            ).tolist()

        effort = gains("effort_limit")
        mg = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        scales = np.array([mg] * 3 + [mg * 0.5] * 3 + effort, dtype=np.float32).tolist()
        dims = dict(
            base_lin_vel=3,
            base_ang_vel=3,
            projected_gravity=3,
            joint_pos=len(names),
            joint_vel=len(names),
            actions=len(names),
            command=3,
        )
        terms = env.observations["actor"].terms
        mode = "none"
        if "momo_disturbance" in terms:
            term = terms["momo_disturbance"]
            func = term.func.__name__
            if func == "momo_dist_effort_scaled_localframe":
                mode = "full"
                dims["momo_disturbance"] = model.nv
                scales[3:6] = np.full(3, mg * term.params["l_ref"], dtype=np.float32).tolist()
            elif func == "zero_momo_leg_effort_scaled_envelope":
                mode = "zero_leg"
                dims["momo_disturbance"] = 14
            else:
                raise ValueError(f"Not a released observer observation: {func}")
        observations = [
            dict(
                name=name,
                dim=dims[name],
                history=int(term.history_length or 1),
                scale=np.asarray(1.0 if term.scale is None else term.scale).tolist(),
            )
            for name, term in terms.items()
        ]
        return cls(
            task,
            names,
            env.sim.mujoco.timestep,
            env.decimation,
            gains("stiffness"),
            gains("damping"),
            gains("armature"),
            effort,
            patterns(robot.init_state.joint_pos),
            patterns(env.actions["joint_pos"].scale, 1.0),
            observations,
            mode,
            scales,
        )


def configure_model(model, cfg):
    """Synchronize PD and armature, returning joint-order -> actuator-order mapping."""
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    if names != cfg.joint_names or (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("The released policies require the supplied 29-DoF G1 joint order")
    actuators = []
    for i, name in enumerate(names):
        joint = model.joint(name)
        ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)
        if len(ids) != 1:
            raise ValueError(f"Expected one position actuator for {name}")
        act = int(ids[0])
        actuators.append(act)
        model.actuator_gainprm[act, 0] = cfg.stiffness[i]
        model.actuator_biasprm[act, 1:3] = [-cfg.stiffness[i], -cfg.damping[i]]
        model.dof_armature[joint.dofadr[0]] = cfg.armature[i]
    model.opt.timestep = cfg.timestep
    return np.asarray(actuators)
