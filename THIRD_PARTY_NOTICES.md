# Source and third-party notices

This document records source provenance and third-party dependencies. It is not a
replacement for upstream licenses, nor a grant of rights to code, assets or model weights.

## ADAPT implementation

Core code and model files were selected from `humanoid-NDO` at source checkout
`4d1cf4bd6b784049751f388cc0fbdb22303af9c8`.

| Released component | Research source |
| --- | --- |
| NumPy observer in `observer/` | `NDO/MoMo.py`, real observer implementation |
| Batched observer and nominal-model environment in `tasks/momo_task/` | `tasks/momo_lumped/momo_env.py` and observer implementation |
| Five training tasks in `tasks/momo_task/` | Selected factories/MDP terms in `tasks/momo_lumped/` |
| Deployment in `deployment/` | Core behavior from `my_deploy/{config_manager,obs_processor,policy_manager,robot_interface}.py` |
| LIO/pelvis adapter in `deployment/` | `my_deploy/{fastlio_receiver_ros2,pelvis_vel_estimator}.py` |
| Models | Exact source files listed in `logs/rsl_rl/4096*1/manifest.json` |
| XML/meshes in `models/g1/` | Referenced closure of the research repository's `g1_real/` |

## Dependencies and assets

- [mjlab](https://github.com/mujocolab/mjlab): training framework and training G1 assets,
  pinned to `18750c1b17ed900842c9824fd403695322c65cdc`.
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): pinned to
  `875c4caf06d71b12b2036c327e7a07ce08a78b9a`.
- [Unitree SDK Python](https://github.com/unitreerobotics/unitree_sdk2_python): robot DDS
  communication; the selected revision is in `pyproject.toml` and `uv.lock`.
  The HG-only CRC implementation is derived from its pure-Python implementation;
  its [BSD-3-Clause license](common/UNITREE_SDK_LICENSE) is retained.
- [Unitree Robotics](https://www.unitree.com/): attribution given by the source G1
  robot-description README. The source `g1_real` directory, released here under
  `models/g1/`, contained no license file.
- MuJoCo, RSL-RL, PyTorch, Warp, ONNX Runtime, CycloneDDS, NumPy, SciPy, Matplotlib
  and the remaining installed dependencies retain their upstream terms and notices.

## Before public redistribution

The repository owner must select the ADAPT code/model license and confirm the original
license and required notices for the redistributed G1 XML/meshes. A project LICENSE
has not yet been selected. Preserve upstream attribution/license files once
their exact applicable sources are confirmed. FAST-LIO source, driver changes, Docker
images and their notices will be handled when that separate component is supplied.
