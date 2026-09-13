"""MDP terms used by the five released training tasks."""

from mjlab.envs.mdp import *
from mjlab.tasks.velocity.mdp.rewards import *
from mjlab.tasks.velocity.mdp.terminations import *
from mjlab.tasks.velocity.mdp.velocity_command import *
from mjlab.tasks.velocity.mdp.curriculums import *
from .dr_curriculum import dr_curriculum
from .observations import (
    LEG_EFFORT_JOINT_NAMES,
    foot_air_time,
    foot_contact,
    foot_contact_forces,
    foot_height,
    momo_dist_effort_scaled_localframe,
    momo_dist_localframe,
    momo_leg_effort_scaled_envelope,
    shared_joint_pos_rel,
    shared_joint_vel_rel,
    zero_momo_leg_effort_scaled_envelope,
)
from .rewards import (
    lin_vel_z_l2,
    momo_leg_cvar_tail,
    momo_leg_peak_guard,
    momo_leg_softstep_reward_schedule,
    track_angular_velocity,
    track_linear_velocity,
)
from .wrench_events import (
    clean_env_curriculum,
    force_cmd_zero_in_settle_window,
    reset_clean_env_flag,
    reset_wrench_state,
    step_continuous_wrenches,
    wrench_event_curriculum,
)
