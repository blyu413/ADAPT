import math

from mjlab.asset_zoo.robots import get_g1_robot_cfg

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab.managers.action_manager import ActionTermCfg

from mjlab.managers.command_manager import CommandTermCfg

from mjlab.managers.event_manager import EventTermCfg

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab.managers.reward_manager import RewardTermCfg

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.scene import SceneCfg

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)

from mjlab.sim import MujocoCfg, SimulationCfg

from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab.envs.mdp import dr

from mjlab.terrains import TerrainEntityCfg

from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.viewer import ViewerConfig

from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from mjlab.asset_zoo.robots import G1_ACTION_SCALE

from mjlab.envs import mdp as envs_mdp

from tasks.momo_task import mdp

MOMO_EFFORT_SCALE_L_REF = 0.5

MOMO_LEG_EFFORT_HISTORY_LENGTH = 40


def _use_effort_scaled_momo_obs(
    cfg: ManagerBasedRlEnvCfg,
    l_ref: float = MOMO_EFFORT_SCALE_L_REF,
    obs_scale: float | None = None,
) -> None:
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms
        replace_kwargs = {
            "func": mdp.momo_dist_effort_scaled_localframe,
            "params": {"asset_cfg": SceneEntityCfg("robot"), "l_ref": l_ref},
        }
        if obs_scale is not None:
            replace_kwargs["scale"] = obs_scale
        terms["momo_disturbance"] = replace(
            terms["momo_disturbance"],
            **replace_kwargs,
        )


def _remove_momo_obs_from_groups(
    cfg: ManagerBasedRlEnvCfg,
    group_names: tuple[str, ...],
) -> None:
    for group_name in group_names:
        cfg.observations[group_name].terms.pop("momo_disturbance", None)


def _use_shared_joint_obs(cfg: ManagerBasedRlEnvCfg) -> None:
    actor_terms = cfg.observations["actor"].terms
    actor_terms["joint_pos"] = replace(
        actor_terms["joint_pos"],
        func=mdp.shared_joint_pos_rel,
        params={},
        noise=None,
        delay_min_lag=0,
        delay_max_lag=0,
    )
    actor_terms["joint_vel"] = replace(
        actor_terms["joint_vel"],
        func=mdp.shared_joint_vel_rel,
        params={},
        noise=None,
        delay_min_lag=0,
        delay_max_lag=0,
    )


def _add_small_joint_friction_rand(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.events["joint_friction_rand"] = EventTermCfg(
        mode="reset",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "abs",
            "ranges": (0.0, 0.12),
        },
    )


def make_momo_lumped_env_cfg() -> ManagerBasedRlEnvCfg:
    """Shared flat-ground locomotion settings for the five G1 tasks.

    The critic adds foot height, air time, contact flags and contact forces.
    Task factories select the residual observations and disturbance curriculum.
    """

    ##
    # Observations
    ##
    history_length = 5
    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=velocity_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=history_length,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=velocity_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=history_length,
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=history_length,
        ),
        "joint_pos": ObservationTermCfg(
            func=velocity_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=history_length,
        ),
        "joint_vel": ObservationTermCfg(
            func=velocity_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=history_length,
        ),
        "actions": ObservationTermCfg(func=velocity_mdp.last_action),
        "command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "twist"},
            history_length=history_length,
        ),
        "momo_disturbance": ObservationTermCfg(
            func=mdp.momo_dist_localframe,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=history_length,
        ),
    }

    critic_terms = {
        **policy_terms,
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    ##
    # Events (domain randomization)
    ##

    events = {
        "reset_base": EventTermCfg(
            func=velocity_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=velocity_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=velocity_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.3, 1.2),
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        # Nominal torso center-of-mass randomization.
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=3.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "lin_vel_z": RewardTermCfg(
            func=mdp.lin_vel_z_l2,
            weight=-2.0,
            params={},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {},  # Set per-robot.
                "std_walking": {},  # Set per-robot.
                "std_running": {},  # Set per-robot.
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=0.0,  # Override per-robot
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=0.0,  # Override per-robot
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.0,  # Override per-robot.
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.5,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
                "height_sensor_name": "foot_height_scan",
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
                "height_sensor_name": "foot_height_scan",
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
    }
    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=velocity_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    ##
    # Curriculum
    ##
    # ================= 1. Curriculum timeline configuration =================
    n_envs = 24

    # --- Velocity curriculum ---
    dur_vel_stage_1 = 5000  # Low-speed warmup (0 -> 5000)
    dur_vel_stage_2 = 5000  # Medium-speed transition (5000 -> 10000)

    # ================= Compute stage boundaries =================
    step_vel_mid = dur_vel_stage_1 * n_envs  # iter 5000
    step_vel_high = (dur_vel_stage_1 + dur_vel_stage_2) * n_envs  # iter 10000

    # ================= 3. Curriculum dictionary =================
    curriculum = {
        # --- 1. Velocity curriculum ---
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    # Stage 1: Low speed (iteration 0)
                    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                    # Stage 2: Medium speed (from iteration 5000)
                    {"step": step_vel_mid, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                    # Stage 3: High speed (from iteration 10000)
                    {"step": step_vel_high, "lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.0, 1.0)},
                ],
            },
        ),
    }
    # ==================================================================================

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="plane",
                max_init_terrain_level=5,
            ),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=100,
                ls_iterations=50,
                ccd_iterations=50,
                integrator="euler",
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )


def unitree_g1_momo_lumped_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the shared G1 flat-ground task, including sensors and joint settings."""
    cfg = make_momo_lumped_env_cfg()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    site_names = ("left_foot", "right_foot")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist.
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,
        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        # Waist.
        r".*waist_yaw.*": 0.3,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.2,
        # Arms.
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        cfg.events.pop("foot_friction", None)
        cfg.events.pop("base_com", None)

        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

    _use_shared_joint_obs(cfg)

    if not play:
        _add_small_joint_friction_rand(cfg)

    if play:
        cfg.commands["twist"] = UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.8, 0.8),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )

    return cfg


def unitree_g1_momo_lumped_flat_effort_scale_env_cfg(
    play: bool = False,
    l_ref: float = MOMO_EFFORT_SCALE_L_REF,
    obs_scale: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Ref-NoMoMo walking config plus physical-scale MoMo residual observations."""
    cfg = unitree_g1_momo_lumped_flat_env_cfg(play=play)
    _use_effort_scaled_momo_obs(cfg, l_ref=l_ref, obs_scale=obs_scale)
    return cfg


def _use_leg_effort_softstep_momo_obs(cfg: ManagerBasedRlEnvCfg) -> None:
    leg_asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=mdp.LEG_EFFORT_JOINT_NAMES,
        preserve_order=True,
    )
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms
        terms["momo_disturbance"] = replace(
            terms["momo_disturbance"],
            func=mdp.momo_leg_effort_scaled_envelope,
            params={"asset_cfg": leg_asset_cfg, "eps": 1e-6},
            history_length=MOMO_LEG_EFFORT_HISTORY_LENGTH,
        )


def _add_momo_leg_softstep_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    leg_asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=mdp.LEG_EFFORT_JOINT_NAMES,
        preserve_order=True,
    )
    common_params = {
        "asset_cfg": leg_asset_cfg,
        "history_length": MOMO_LEG_EFFORT_HISTORY_LENGTH,
        "command_name": "twist",
        "command_threshold": 0.05,
        "eps": 1e-6,
    }
    cfg.rewards["momo_leg_cvar_tail"] = RewardTermCfg(
        func=mdp.momo_leg_cvar_tail,
        weight=-0.05,
        params={
            **common_params,
            "threshold": 0.145,
            "margin": 0.020,
            "quantile": 0.9,
        },
    )
    cfg.rewards["momo_leg_peak_guard"] = RewardTermCfg(
        func=mdp.momo_leg_peak_guard,
        weight=-0.02,
        params={
            **common_params,
            "threshold": 0.180,
            "margin": 0.030,
        },
    )

    n_envs = 24
    step_vel_mid = 5000 * n_envs
    step_vel_high = 10000 * n_envs
    cfg.curriculum["momo_leg_softstep_reward"] = CurriculumTermCfg(
        func=mdp.momo_leg_softstep_reward_schedule,
        params={
            "stages": [
                {
                    "step": 0,
                    "cvar_threshold": 0.145,
                    "cvar_margin": 0.020,
                    "cvar_weight": -0.05,
                    "peak_threshold": 0.180,
                    "peak_margin": 0.030,
                    "peak_weight": -0.02,
                },
                {
                    "step": step_vel_mid,
                    "cvar_threshold": 0.132,
                    "cvar_margin": 0.015,
                    "cvar_weight": -0.12,
                    "peak_threshold": 0.165,
                    "peak_margin": 0.025,
                    "peak_weight": -0.05,
                },
                {
                    "step": step_vel_high,
                    "cvar_threshold": 0.120,
                    "cvar_margin": 0.012,
                    "cvar_weight": -0.25,
                    "peak_threshold": 0.150,
                    "peak_margin": 0.020,
                    "peak_weight": -0.08,
                },
            ],
        },
    )


def unitree_g1_momo_lumped_flat_leg_effort_softstep_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """EffortScale walker with leg-only MoMo envelope observation and rewards."""
    cfg = unitree_g1_momo_lumped_flat_effort_scale_env_cfg(play=play)
    _use_leg_effort_softstep_momo_obs(cfg)
    _add_momo_leg_softstep_rewards(cfg)
    return cfg


def unitree_g1_momo_lumped_flat_leg_effort_softstep_no_actor_feedback_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """LegEffortSoftStep control with actor MoMo feedback zeroed out."""
    cfg = unitree_g1_momo_lumped_flat_leg_effort_softstep_env_cfg(play=play)
    actor_terms = cfg.observations["actor"].terms
    actor_terms["momo_disturbance"] = replace(
        actor_terms["momo_disturbance"],
        func=mdp.zero_momo_leg_effort_scaled_envelope,
    )
    return cfg


def unitree_g1_momo_lumped_flat_force_env_cfg(
    play: bool = False,
    dur_warmup: int = 0,
    dur_static_intro: int = 5000,
    dur_static_hold: int = 5000,
    dur_impulse_intro: int = 5000,
) -> ManagerBasedRlEnvCfg:
    """Torso force/torque curriculum for continuation from a walking checkpoint.

    ``dur_warmup`` offsets the disturbance schedule against the restored training
    step. Velocity commands use the full Stage 2 range without a velocity ramp.
    """
    cfg = unitree_g1_momo_lumped_flat_env_cfg(play=play)

    if play:
        return cfg

    # ================= Curriculum timeline configuration =================
    n_envs = 24  # = num_steps_per_env

    step_dr_offset = 0

    step_static_start = step_dr_offset + dur_warmup * n_envs
    step_static_max = step_static_start + dur_static_intro * n_envs
    step_impulse_start = step_static_max + dur_static_hold * n_envs
    step_impulse_max = step_impulse_start + dur_impulse_intro * n_envs

    # ================= Wrench reset events =================
    for state_attr in [
        "wrench_state_static_force",
        "wrench_state_static_torque",
        "wrench_state_impulse_force",
        "wrench_state_impulse_torque",
    ]:
        key = f"wrench_reset_{state_attr.replace('wrench_state_', '')}"
        cfg.events[key] = EventTermCfg(
            func=mdp.reset_wrench_state,
            mode="reset",
            params={"state_attr": state_attr},
        )

    # ================= Wrench curriculum =================
    cfg.curriculum["wrench_static_force"] = CurriculumTermCfg(
        func=mdp.wrench_event_curriculum,
        params={
            "state_attr": "wrench_state_static_force",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "force_stages": [
                {
                    "step": step_dr_offset,
                    "range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
                {
                    "step": step_static_start,
                    "range": {"x": (-20.0, 20.0), "y": (-20.0, 20.0), "z": (-20.0, 10.0)},
                },
                {
                    "step": step_static_max,
                    "range": {"x": (-40.0, 40.0), "y": (-40.0, 40.0), "z": (-40.0, 10.0)},
                },
            ],
            "duration_stages": [
                {"step": step_dr_offset, "range": (0.0, 1.0)},
                {"step": step_static_start, "range": (5.0, 7.0)},
                {"step": step_static_max, "range": (6.0, 10.0)},
            ],
            "idle_stages": [{"step": step_dr_offset, "range": (4.0, 7.0)}],
        },
    )
    cfg.curriculum["wrench_static_torque"] = CurriculumTermCfg(
        func=mdp.wrench_event_curriculum,
        params={
            "state_attr": "wrench_state_static_torque",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "torque_stages": [
                {
                    "step": step_dr_offset,
                    "range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
                {
                    "step": step_static_start,
                    "range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (-2.5, 2.5)},
                },
                {
                    "step": step_static_max,
                    "range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (-5.0, 5.0)},
                },
            ],
            "duration_stages": [
                {"step": step_static_start, "range": (3.0, 6.0)},
                {"step": step_static_max, "range": (6.0, 10.0)},
            ],
            "idle_stages": [{"step": step_dr_offset, "range": (4.0, 7.0)}],
        },
    )
    cfg.curriculum["wrench_impulse_force"] = CurriculumTermCfg(
        func=mdp.wrench_event_curriculum,
        params={
            "state_attr": "wrench_state_impulse_force",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "force_stages": [
                {
                    "step": step_dr_offset,
                    "range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
                {
                    "step": step_impulse_start,
                    "range": {"x": (-40.0, 40.0), "y": (-40.0, 40.0), "z": (-20.0, 10.0)},
                },
                {
                    "step": step_impulse_max,
                    "range": {"x": (-80.0, 80.0), "y": (-80.0, 80.0), "z": (-20.0, 10.0)},
                },
            ],
            "duration_stages": [{"step": step_dr_offset, "range": (0.1, 0.2)}],
            "idle_stages": [
                {"step": step_impulse_start, "range": (8.0, 10.0)},
                {"step": step_impulse_max, "range": (4.0, 6.0)},
            ],
        },
    )
    cfg.curriculum["wrench_impulse_torque"] = CurriculumTermCfg(
        func=mdp.wrench_event_curriculum,
        params={
            "state_attr": "wrench_state_impulse_torque",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "torque_stages": [
                {
                    "step": step_dr_offset,
                    "range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
                {
                    "step": step_impulse_start,
                    "range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (-5.0, 5.0)},
                },
                {
                    "step": step_impulse_max,
                    "range": {"x": (-20.0, 20.0), "y": (-20.0, 20.0), "z": (-10.0, 10.0)},
                },
            ],
            "duration_stages": [{"step": step_dr_offset, "range": (0.1, 0.2)}],
            "idle_stages": [
                {"step": step_impulse_start, "range": (8.0, 10.0)},
                {"step": step_impulse_max, "range": (4.0, 6.0)},
            ],
        },
    )

    # Stage 2 continues a trained walker, so use the full command range.
    cfg.commands["twist"].ranges.lin_vel_x = (-1.5, 1.5)
    cfg.commands["twist"].ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    cfg.curriculum.pop("command_vel", None)

    return cfg


def unitree_g1_momo_lumped_flat_hand_load_env_cfg(
    play: bool = False,
    dur_warmup: int = 0,
    dur_static_intro: int = 5000,
    dur_static_hold: int = 5000,
    dur_impulse_intro: int = 5000,
) -> ManagerBasedRlEnvCfg:
    """Add wrist loads and hardware randomization to the torso-force task.

    Wrist force vectors are sampled independently from the same ranges, with
    one shared force/idle timer per environment. Final vertical load is 0-24 N
    downward per wrist.
    """
    cfg = unitree_g1_momo_lumped_flat_force_env_cfg(
        play=play,
        dur_warmup=dur_warmup,
        dur_static_intro=dur_static_intro,
        dur_static_hold=dur_static_hold,
        dur_impulse_intro=dur_impulse_intro,
    )

    # Share the same joint encoder readout between policy and MoMo: actor
    # reads MoMoEnv's noisy ideal_qpos/ideal_qvel, matching real-robot semantics
    # (one encoder reading, two consumers). Drop per-term Unoise to avoid
    # double-adding noise on top of the already-noisy shared readout.
    for group_name, term_func in (
        ("joint_pos", mdp.shared_joint_pos_rel),
        ("joint_vel", mdp.shared_joint_vel_rel),
    ):
        cfg.observations["actor"].terms[group_name] = replace(
            cfg.observations["actor"].terms[group_name],
            func=term_func,
            params={},
            noise=None,
        )

    if play:
        return cfg

    n_envs = 24

    # ================= Hand load curriculum =================
    # Use the same stage boundaries as torso static forces, with step_dr_offset=0.
    step_dr_offset = 0
    step_static_start = step_dr_offset + dur_warmup * n_envs
    step_static_max = step_static_start + dur_static_intro * n_envs

    # Wrench reset event for hand load
    cfg.events["wrench_reset_hand_load"] = EventTermCfg(
        func=mdp.reset_wrench_state,
        mode="reset",
        params={"state_attr": "wrench_state_hand_load"},
    )

    # Curriculum: force at wrists (simulate carrying objects)
    # Per wrist: X/Y up to ±8 N, Z in [-24, 0] N.
    cfg.curriculum["wrench_hand_load"] = CurriculumTermCfg(
        func=mdp.wrench_event_curriculum,
        params={
            "state_attr": "wrench_state_hand_load",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")
            ),
            "force_stages": [
                {
                    "step": step_dr_offset,
                    "range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
                {
                    "step": step_static_start,
                    "range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0), "z": (-12.0, 0.0)},
                },
                {
                    "step": step_static_max,
                    "range": {"x": (-8.0, 8.0), "y": (-8.0, 8.0), "z": (-24.0, 0.0)},
                },
            ],
            # Force and idle durations match the torso static-force schedule.
            "duration_stages": [
                {"step": step_dr_offset, "range": (0.0, 1.0)},
                {"step": step_static_start, "range": (5.0, 7.0)},
                {"step": step_static_max, "range": (6.0, 10.0)},
            ],
            "idle_stages": [{"step": step_dr_offset, "range": (4.0, 7.0)}],
        },
    )

    # ============ Hardware DR events (reset mode, curriculum-controlled ranges) ============
    LEG_REGEX = ".*hip.*|.*knee.*|.*ankle.*"
    WAIST_REGEX = ".*waist.*"
    ARM_REGEX = ".*shoulder.*|.*elbow.*|.*wrist.*"

    # ============ Body mass/inertia DR (startup: sampled once per env) ============
    # pseudo_inertia's alpha_range scales mass and inertia consistently through density,
    # avoiding the inconsistency of changing mass alone with body_mass. ±0.05 ≈ ±5% density.
    PI_LEGS_BODIES = (".*hip.*", ".*knee.*", ".*ankle.*")
    PI_TORSO_BODIES = ("pelvis", "torso_link", ".*waist.*")
    PI_ARMS_BODIES = (".*shoulder.*", ".*elbow.*", ".*wrist.*")
    for tag, body_pat in [
        ("legs", PI_LEGS_BODIES),
        ("torso", PI_TORSO_BODIES),
        ("arms", PI_ARMS_BODIES),
    ]:
        cfg.events[f"pseudo_inertia_rand_{tag}"] = EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=body_pat),
                "alpha_range": (-0.05, 0.05),
            },
        )

    # friction: abs mode (G1 default=0, so init (0,0) is identity)
    cfg.events["joint_friction_rand"] = EventTermCfg(
        mode="reset",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "abs",
            "ranges": {LEG_REGEX: (0.0, 0.0), WAIST_REGEX: (0.0, 0.0), ARM_REGEX: (0.0, 0.0)},
        },
    )
    # damping: scale mode (init x1.0 identity; G1 default damping varies per joint group)
    cfg.events["joint_damping_rand"] = EventTermCfg(
        mode="reset",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": {LEG_REGEX: (1.0, 1.0), WAIST_REGEX: (1.0, 1.0), ARM_REGEX: (1.0, 1.0)},
        },
    )
    # ============ Hardware DR curriculum ============
    # Align the timeline with the wrench schedule using the stage boundaries above.
    step_dr_start = step_static_start
    step_dr_mid = (step_static_start + step_static_max) // 2
    step_dr_max = step_static_max

    cfg.curriculum["hardware_dr_schedule"] = CurriculumTermCfg(
        func=mdp.dr_curriculum,
        params={
            "stages": [
                # joint_friction_rand (abs; G1 default=0)
                {
                    "step": step_dr_start,
                    "event_name": "joint_friction_rand",
                    "ranges": {
                        LEG_REGEX: (0.0, 0.0),
                        WAIST_REGEX: (0.0, 0.0),
                        ARM_REGEX: (0.0, 0.0),
                    },
                },
                {
                    "step": step_dr_mid,
                    "event_name": "joint_friction_rand",
                    "ranges": {
                        LEG_REGEX: (0.05, 0.4),
                        WAIST_REGEX: (0.05, 0.25),
                        ARM_REGEX: (0.05, 0.2),
                    },
                },
                {
                    "step": step_dr_max,
                    "event_name": "joint_friction_rand",
                    "ranges": {
                        LEG_REGEX: (0.05, 1.5),
                        WAIST_REGEX: (0.05, 1.0),
                        ARM_REGEX: (0.05, 0.8),
                    },
                },
                # joint_damping_rand (scale; init x1.0 identity)
                {
                    "step": step_dr_start,
                    "event_name": "joint_damping_rand",
                    "ranges": {
                        LEG_REGEX: (1.0, 1.0),
                        WAIST_REGEX: (1.0, 1.0),
                        ARM_REGEX: (1.0, 1.0),
                    },
                },
                {
                    "step": step_dr_mid,
                    "event_name": "joint_damping_rand",
                    "ranges": {
                        LEG_REGEX: (0.7, 1.3),
                        WAIST_REGEX: (0.8, 1.2),
                        ARM_REGEX: (0.8, 1.2),
                    },
                },
                {
                    "step": step_dr_max,
                    "event_name": "joint_damping_rand",
                    "ranges": {
                        LEG_REGEX: (0.5, 1.5),
                        WAIST_REGEX: (0.7, 1.3),
                        ARM_REGEX: (0.7, 1.3),
                    },
                },
            ],
        },
    )

    return cfg


def unitree_g1_momo_lumped_flat_hand_load_clean_mix_env_cfg(
    play: bool = False,
    dur_warmup: int = 0,
    dur_static_intro: int = 5000,
    dur_static_hold: int = 5000,
    dur_impulse_intro: int = 5000,
    clean_prob_max: float = 0.3,
    settle_duration_s: float = 5.0,
) -> ManagerBasedRlEnvCfg:
    """HandLoad with a settling window and clean episodes.

    Every episode starts with ``settle_duration_s`` of zero velocity commands;
    this does not change the wrench timers. At reset, the clean-probability
    curriculum selects environments whose five wrench sources remain masked
    for the whole episode (default target probability: 0.3).
    """
    cfg = unitree_g1_momo_lumped_flat_hand_load_env_cfg(
        play=play,
        dur_warmup=dur_warmup,
        dur_static_intro=dur_static_intro,
        dur_static_hold=dur_static_hold,
        dur_impulse_intro=dur_impulse_intro,
    )

    if play:
        return cfg

    n_envs = 24
    step_dr_offset = 0
    step_static_start = step_dr_offset + dur_warmup * n_envs

    # M1: Force cmd=0 for the first settle_duration_s of each episode.
    cfg.events["force_cmd_zero_settle"] = EventTermCfg(
        func=mdp.force_cmd_zero_in_settle_window,
        mode="step",
        params={
            "command_term_name": "twist",
            "settle_duration_s": settle_duration_s,
        },
    )

    # M2: Resample the clean flag from Bernoulli(clean_prob) at each reset.
    cfg.events["wrench_clean_env_reset"] = EventTermCfg(
        func=mdp.reset_clean_env_flag,
        mode="reset",
        params={
            "flag_attr": "wrench_clean_env_flag",
            "clean_prob_attr": "wrench_clean_env_prob",
        },
    )

    # Curriculum: Step clean_prob from 0 to clean_prob_max at step_static_start.
    cfg.curriculum["wrench_clean_env_prob"] = CurriculumTermCfg(
        func=mdp.clean_env_curriculum,
        params={
            "clean_prob_attr": "wrench_clean_env_prob",
            "prob_stages": [
                {"step": step_dr_offset, "value": 0.0},
                {"step": step_static_start, "value": clean_prob_max},
            ],
        },
    )

    return cfg


def unitree_g1_momo_lumped_flat_hand_load_clean_mix_effort_scale_env_cfg(
    play: bool = False,
    dur_warmup: int = 15000,
    dur_static_intro: int = 5000,
    dur_static_hold: int = 5000,
    dur_impulse_intro: int = 5000,
    clean_prob_max: float = 0.3,
    settle_duration_s: float = 5.0,
    l_ref: float = MOMO_EFFORT_SCALE_L_REF,
    obs_scale: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """HandLoad-CleanMix task with physical-scale MoMo residual observations."""
    cfg = unitree_g1_momo_lumped_flat_hand_load_clean_mix_env_cfg(
        play=play,
        dur_warmup=dur_warmup,
        dur_static_intro=dur_static_intro,
        dur_static_hold=dur_static_hold,
        dur_impulse_intro=dur_impulse_intro,
        clean_prob_max=clean_prob_max,
        settle_duration_s=settle_duration_s,
    )
    _use_effort_scaled_momo_obs(cfg, l_ref=l_ref, obs_scale=obs_scale)
    return cfg


def _remove_momo_disturbance_obs(cfg: ManagerBasedRlEnvCfg) -> None:
    """Remove momo_disturbance from policy and critic observations (bypass mode)."""
    _remove_momo_obs_from_groups(cfg, ("actor", "critic"))


def unitree_g1_momo_lumped_flat_ref_nomomo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Stage 1 walking baseline without actor or critic residual observations."""
    cfg = unitree_g1_momo_lumped_flat_env_cfg(play=play)
    _remove_momo_disturbance_obs(cfg)
    return cfg


def unitree_g1_momo_lumped_flat_ref_nomomo_hand_load_clean_mix_env_cfg(
    play: bool = False,
    dur_warmup: int = 15000,
    dur_static_intro: int = 5000,
    dur_static_hold: int = 5000,
    dur_impulse_intro: int = 5000,
    clean_prob_max: float = 0.3,
    settle_duration_s: float = 5.0,
) -> ManagerBasedRlEnvCfg:
    """Stage 2 CleanMix baseline without actor or critic residual observations.

    The default 15,000-iteration offset aligns the disturbance schedule with
    the restored Stage 1 step counter, so continuation starts at the low-load
    stage. Settling, clean episodes and hardware randomization match ADAPT.
    """
    cfg = unitree_g1_momo_lumped_flat_hand_load_clean_mix_env_cfg(
        play=play,
        dur_warmup=dur_warmup,
        dur_static_intro=dur_static_intro,
        dur_static_hold=dur_static_hold,
        dur_impulse_intro=dur_impulse_intro,
        clean_prob_max=clean_prob_max,
        settle_duration_s=settle_duration_s,
    )
    _remove_momo_disturbance_obs(cfg)
    return cfg
