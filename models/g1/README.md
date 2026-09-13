# G1 deployment assets

This is the reference closure of the research repository's `g1_real` deployment
assets: three XML files and the 36 meshes referenced by them. Shared meshes are
stored once under `models/g1/`. Physical model definitions and meshes are retained;
obsolete commented-out sensor examples have been removed.

- `scene.xml` includes `g1_29dof_modified.xml`: flat-ground MuJoCo deployment.
- `g1_29dof.xml`: nominal real-robot observer and pelvis/LiDAR kinematics.
- Training uses the G1 asset supplied by the pinned mjlab dependency.

Deployment overwrites position actuator gains, damping and joint armature using
the selected model's JSON, matching the research deployment. Do not substitute
an arbitrary G1 asset: joint order, inertias and actuator settings affect both
the policy and observer. The real-control adapter accepts the original 29-DoF
hardware `mode_machine=2`; revised hip reductions require a separately validated asset/config.

The source asset README attributes the robot description to Unitree Robotics.
No asset license file was present in the copied research directory. See
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) before redistribution.
