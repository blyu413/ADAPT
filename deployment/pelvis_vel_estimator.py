"""
Pelvis Velocity Estimator — LiDAR-local velocity → pelvis-local velocity

Purpose:
  Use MuJoCo kinematics (Jacobians) to convert velocity in the LiDAR
  (mid360_link) local frame to velocity in the pelvis local frame.

Derivation overview (see the estimate() docstring for details):
  Kinematic chain: pelvis → waist_yaw → waist_roll → waist_pitch
                  (torso_link) → [fixed] → mid360_link (LiDAR)

  Fixed mid360 transform relative to torso_link (from URDF):
    offset = [0.0002835, 0.00003, 0.40618]
    rpy    = [0, 0.04014257279586953, 0]

  Extra rotation of the FAST-LIO output frame relative to the URDF frame:
    Livox driver compensates for inverted mounting: lidar_mount_extra_rpy=[0,0,0]
    Output uses the raw sensor frame: lidar_mount_extra_rpy=[π,0,0]
    R_actual = R_mid360_urdf @ R_extra

Design decisions:
  - mid360_link is absent from the MuJoCo XML (lost in URDF→MJCF conversion).
  - Compute its pose from torso_link and the known fixed offset instead.
  - Set the pelvis to the identity pose (pos=0, quat=[1,0,0,0]) in the internal
    mjData so the MuJoCo world frame equals the pelvis local frame.

Example:
    import mujoco
    model = mujoco.MjModel.from_xml_path("models/g1/g1_29dof.xml")
    estimator = PelvisVelocityEstimator(model)

    v_pelvis, omega_pelvis = estimator.estimate(
        q_joints, dq_joints, v_lidar_local, omega_lidar_local
    )
"""

import numpy as np
import mujoco

# =================================================================
#  Fixed mid360_link transform relative to torso_link (from URDF)
# =================================================================
# <joint name="mid360_joint" type="fixed">
#   <origin xyz="0.0002835 0.00003 0.40618" rpy="0 0.04014257279586953 0"/>
#   <parent link="torso_link"/>
#   <child link="mid360_link"/>
# </joint>

MID360_OFFSET_POS = np.array([0.0002835, 0.00003, 0.40618], dtype=np.float64)
MID360_OFFSET_RPY = np.array([0.0, 0.04014257279586953, 0.0], dtype=np.float64)

# =================================================================
#  LiDAR mounting correction — FAST-LIO output frame relative to URDF mid360_link
# =================================================================
# Depends on whether the Livox driver compensates for inverted mounting:
#   - Compensated (extrinsic roll=180°): output is effectively upright → [0, 0, 0]
#   - Uncompensated (raw sensor frame): apply mounting correction here → [π, 0, 0]
LIDAR_MOUNT_EXTRA_RPY_DEFAULT = np.array([np.pi, 0.0, 0.0], dtype=np.float64)


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """
    Convert RPY (roll-pitch-yaw) Euler angles to a rotation matrix.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (URDF / ROS standard).

    Args:
        rpy: (3,) [roll, pitch, yaw] in radians

    Returns:
        (3, 3) rotation matrix
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


class PelvisVelocityEstimator:
    """
    Convert mid360 (LiDAR) local-frame velocity to pelvis local-frame velocity.

    Method:
      1. Place the pelvis at the origin with identity orientation in the internal
         mjData, making the MuJoCo world frame equal to the pelvis local frame.
      2. Set joint angles q and call mj_step1 to obtain body poses and Jacobian state.
      3. Use torso_link and the fixed offset to obtain the mid360 pose in the pelvis frame.
      4. Use mj_jac for the mid360 point rigidly attached to torso_link.
      5. Given v_mid360_pelvis = R_mid360 @ v_lidar_local, use the Jacobian to solve
         for v_root (= v_pelvis_pelvis, since the pelvis is at the origin).
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        torso_body_name: str = "torso_link",
        pelvis_body_name: str = "pelvis",
        lidar_mount_extra_rpy: np.ndarray | None = None,
    ):
        """
        Args:
            mj_model: Loaded MuJoCo model; must include a floating-base joint.
            torso_body_name: Torso body name, without a prefix.
            pelvis_body_name: Pelvis body name, without a prefix.
            lidar_mount_extra_rpy: Extra RPY rotation (rad) of the FAST-LIO output
                frame relative to URDF mid360_link. Pass [0,0,0] if the Livox driver
                compensates for inverted mounting (extrinsic roll=180°), or
                [π,0,0] (default) if FAST-LIO outputs in the raw sensor frame.
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)  # Independent data; leaves external viewers unchanged.

        # Resolve body IDs.
        self.torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
        self.pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, pelvis_body_name)
        if self.torso_id < 0:
            raise ValueError(f"Body '{torso_body_name}' not found in MuJoCo model")
        if self.pelvis_id < 0:
            raise ValueError(f"Body '{pelvis_body_name}' not found in MuJoCo model")

        # Model dimensions.
        self.nv = mj_model.nv
        self.nq = mj_model.nq
        self.num_joints = self.nv - 6  # Exclude the 6-DoF floating base.

        # Precompute the fixed mid360-to-torso transform defined in the URDF.
        self._p_mid360_torso = MID360_OFFSET_POS.copy()
        self._R_mid360_torso = rpy_to_rotation_matrix(MID360_OFFSET_RPY)  # URDF mid360_link frame

        # Precompute the extra LiDAR mounting rotation.
        mount_rpy = (
            np.asarray(lidar_mount_extra_rpy, dtype=np.float64)
            if lidar_mount_extra_rpy is not None
            else LIDAR_MOUNT_EXTRA_RPY_DEFAULT
        )
        self._R_mount_extra = rpy_to_rotation_matrix(mount_rpy)

        # Compose: R_lidar_actual_rel_torso = R_mid360_urdf_rel_torso @ R_extra.
        # This rotates the FAST-LIO velocity output frame into the torso frame.
        self._R_lidar_actual_torso = self._R_mid360_torso @ self._R_mount_extra

        # Preallocate Jacobian buffers.
        self._jacp = np.zeros((3, self.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.nv), dtype=np.float64)

        print("[PelvisVelEstimator] Initialization complete")
        print(f"  torso body: '{torso_body_name}' (id={self.torso_id})")
        print(f"  pelvis body: '{pelvis_body_name}' (id={self.pelvis_id})")
        print(f"  nv={self.nv}, num_joints={self.num_joints}")
        print(f"  mid360 offset (torso frame): {self._p_mid360_torso}")
        print(f"  lidar mount extra RPY (rad): {mount_rpy}")

    def estimate(
        self,
        q_joints: np.ndarray,
        dq_joints: np.ndarray,
        v_lidar_local: np.ndarray,
        omega_lidar_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate pelvis linear and angular velocity in its local frame.

        Derivation:
        ───────────
        Set the pelvis to the identity pose in the internal data, so the MuJoCo
        world frame equals the pelvis frame.

        1) Forward kinematics: q → R_torso, p_torso (torso pose in the pelvis frame)
           R_lidar_actual = R_torso @ R_mid360_rel @ R_mount_extra
           p_mid360 = p_torso + R_torso @ offset

        2) Transform LiDAR-local velocity into the pelvis frame (= MuJoCo world frame):
           FAST-LIO velocities use the actual LiDAR frame, including the mounting rotation.
           v_mid360_pf = R_lidar_actual @ v_lidar_local
           ω_mid360_pf = R_lidar_actual @ omega_lidar_local

        3) Decompose the MuJoCo Jacobian with the pelvis at the origin:

           mj_jac(point=p_mid360, body=torso) → J_lin (3×nv), J_rot (3×nv)

           Structure (the free joint occupies the first 6 columns):
             J_lin[:,0:3] = I₃       (root translation moves all points equally)
             J_rot[:,0:3] = 0        (root translation does not affect angular velocity)
             J_rot[:,3:6] = I₃       (root angular velocity is transmitted unchanged)

           Therefore:
             ω_mid360_pf = ω_root + J_rot[:,6:] @ dq        ...(A)
             v_mid360_pf = v_root + J_lin[:,3:6] @ ω_root
                         + J_lin[:,6:] @ dq                  ...(B)

        4) Solve (A) for ω_root (= ω_pelvis_pelvis):
             ω_root = ω_mid360_pf − J_rot[:,6:] @ dq

        5) Solve (B) for v_root (= v_pelvis_pelvis):
             v_root = v_mid360_pf − J_lin[:,3:6] @ ω_root − J_lin[:,6:] @ dq

        Args:
            q_joints:          (num_joints,) joint positions (rad)
            dq_joints:         (num_joints,) joint velocities (rad/s)
            v_lidar_local:     (3,) LiDAR-local linear velocity (m/s)
            omega_lidar_local: (3,) LiDAR-local angular velocity (rad/s)

        Returns:
            v_pelvis_local:     (3,) pelvis-local linear velocity (m/s)
            omega_pelvis_local: (3,) pelvis-local angular velocity (rad/s)
        """
        # --- Step 1: Set the pelvis to the identity pose and fill in joint angles. ---
        self.data.qpos[:3] = 0.0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quat [w, x, y, z]
        nj = min(len(q_joints), self.num_joints)
        self.data.qpos[7 : 7 + nj] = q_joints[:nj]

        # --- Step 2: Forward kinematics (only body poses and joint axes are needed). ---
        # Use mj_step1 instead of mj_forward to skip constraint solving (contact/friction)
        # while initializing the xpos/xmat/xaxis/xanchor state required by mj_jac.
        # mj_kinematics alone does not initialize xaxis/xanchor, leading to incorrect Jacobians.
        mujoco.mj_step1(self.model, self.data)

        # --- Step 3: Compute the mid360 pose in the pelvis frame. ---
        R_torso = self.data.xmat[self.torso_id].reshape(3, 3)
        p_torso = self.data.xpos[self.torso_id]

        # mid360 position depends only on the fixed URDF offset, not the mounting orientation.
        p_mid360 = p_torso + R_torso @ self._p_mid360_torso

        # Actual LiDAR frame used for FAST-LIO velocity output.
        # R_lidar_actual = R_torso @ R_mid360_urdf_rel @ R_mount_extra
        R_lidar_actual = R_torso @ self._R_lidar_actual_torso

        # --- Step 4: Transform LiDAR-local velocities into the pelvis frame. ---
        # v_lidar_local / omega_lidar_local are FAST-LIO outputs in the actual LiDAR frame.
        v_mid360_pf = R_lidar_actual @ v_lidar_local
        omega_mid360_pf = R_lidar_actual @ omega_lidar_local

        # --- Step 5: Compute the Jacobian at the mid360 point attached to the torso. ---
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jac(
            self.model,
            self.data,
            self._jacp,
            self._jacr,
            p_mid360,  # point in world coordinates
            self.torso_id,  # body it's attached to
        )

        # --- Step 6: Extract joint Jacobian columns, skipping the 6 free-joint columns. ---
        J_lin_joints = self._jacp[:, 6:]  # (3, num_joints)
        J_rot_joints = self._jacr[:, 6:]  # (3, num_joints)
        J_lin_omega = self._jacp[:, 3:6]  # (3, 3) — root rotation contribution to linear velocity

        dq = np.zeros(self.num_joints, dtype=np.float64)
        dq[:nj] = dq_joints[:nj]

        # --- Step 7: Solve for ω_root (pelvis angular velocity in the pelvis frame). ---
        #   ω_mid360_pf = ω_root + J_rot_joints @ dq
        omega_root = omega_mid360_pf - J_rot_joints @ dq

        # --- Step 8: Solve for v_root (pelvis linear velocity in the pelvis frame). ---
        #   v_mid360_pf = v_root + J_lin_omega @ ω_root + J_lin_joints @ dq
        v_root = v_mid360_pf - J_lin_omega @ omega_root - J_lin_joints @ dq

        return v_root, omega_root
