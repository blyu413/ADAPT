# ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion

Project page: [ADAPT](https://blyu413.github.io/adapt-locomotion/)

Paper: [arXiv:2606.16542](https://arxiv.org/abs/2606.16542)

![ADAPT overview](images/overview.png)

## Abstract

Humanoids deployed in human-centered environments must handle force-interactive tasks, where external contacts introduce unexpected disturbances that disrupt locomotion accuracy and stability. Existing learning-based approaches rely on broad domain randomization, task-specific force objectives, or learning-based force estimators from motion history, each of which compromises accuracy, task transferability, or out-of-distribution robustness. We present Analytical Disturbance-Aware Policy Training (ADAPT), a framework that equips humanoid policies with a physically grounded disturbance observer. ADAPT estimates residual force/torque online from accessible robot dynamics without requiring force/torque sensors, feeds these estimates directly into the policy, and uses them as a physics-derived signal for robust behavior under external forces, payloads, and contact-induced disturbances. Experiments on a Unitree G1 humanoid show accurate disturbance prediction, stronger robustness than a proprioception-only baseline, improved velocity tracking under out-of-distribution disturbances, and the ability to encourage lighter locomotion by penalizing inferred lower-body disturbances.

## Project Structure

The main directories and files are organized as follows:

```text
ADAPT/
├── common/         # Shared utilities for training and deployment
├── deployment/     # MuJoCo and Unitree G1 deployment code
├── images/
├── logs/rsl_rl/    # Released checkpoints and training outputs
├── models/g1/      # G1 MuJoCo models and mesh assets
├── observer/       # Momentum observer implementation
├── scripts/        # Training, playback, export, and monitoring scripts
├── tasks/          # MJLab task environments and configurations
├── pyproject.toml  # Project metadata, dependencies, and commands
└── uv.lock         # Locked dependency versions
```

## Installation

ADAPT uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

### Requirements

| Item | Details |
| --- | --- |
| Platform | Linux (Ubuntu 22.04) |
| Python | 3.10 |
| CUDA | 12.8 |

### 1. Install `uv`

Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/blyu413/ADAPT.git
cd ADAPT
```

### 3. Install the dependencies for your use case

```bash
uv sync --frozen
```

The following optional dependency groups are available:

| Use case | Command | Main additions |
| --- | --- | --- |
| Train, play, or export a PyTorch policy | `uv sync --frozen --extra train` | mjlab, PyTorch 2.9.1 + CUDA 12.8, MuJoCo Warp, RSL-RL |
| Use a controller in MuJoCo | `uv sync --frozen --extra controller` | pygame |
| Connect to a Unitree G1 | `uv sync --frozen --extra robot` | Unitree SDK2, CycloneDDS, pygame |
| Plot live or recorded telemetry | `uv sync --frozen --extra monitor` | Matplotlib |
| Install everything | `uv sync --frozen --all-extras` | All of the above |

Extras can be combined, for example:

```bash
uv sync --frozen --extra train --extra monitor
```

`uv sync` creates `.venv/` and performs an exact sync. When running it again,
include every extra that should remain installed. Use `uv run --frozen ...` for
the commands in this repository; activating the environment manually is
optional.

### 4. Verify the installation

Run the released ADAPT ONNX policy in MuJoCo:

```bash
uv run --frozen adapt-sim --seconds 10
```

Use the `--headless` option when running without a display.

## Train & Deploy

### Tasks

| Task name | Description |
| --- | --- |
| `MOMO-Lumped-Flat-Unitree-G1-EffortScale` | ADAPT Stage 1 locomotion training |
| `MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale` | ADAPT Stage 2 hand-load training |
| `Ref-NoMoMo-Flat-Unitree-G1` | Proprioception-only baseline Stage 1 |
| `Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix` | Proprioception-only baseline Stage 2 |
| `MOMO-Lumped-Flat-Unitree-G1-LegEffortSoftStep-NoActorFeedback` | Light-step training without observer feedback to the actor |

### Training

Run the two-stage ADAPT training pipeline:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --num-envs 4096 \
  --stage1-iterations 15000 \
  --stage2-iterations 20000 \
  --log-root logs/rsl_rl
```

The first stage trains the locomotion policy, and the second stage continues
training with randomized hand loads. The final Stage 1 checkpoint is loaded
automatically before Stage 2 begins.

ADAPT retains MJLab's multi-GPU launcher. Select multiple GPUs with a quoted
list; `--num-envs` is the number of environments on each GPU:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --gpu-ids "[0, 1]" \
  --num-envs 4096
```

TensorBoard is the offline-safe default. To log both stages to Weights & Biases,
select the W&B logger explicitly:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --logger wandb \
  --wandb-project adapt
```

Add `--upload-model` if checkpoints and exported models should also be uploaded.

The two stages can also be trained separately. Train Stage 1 with:

```bash
uv run --extra train python scripts/train.py \
  MOMO-Lumped-Flat-Unitree-G1-EffortScale \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 15000
```

The single-task entry point accepts MJLab's `--gpu-ids` and RSL-RL logger
configuration directly, for example `--gpu-ids "[0, 1]"` or
`--agent.logger wandb --agent.wandb-project adapt`.

Then continue with the Stage 2 hand-load task:

```bash
uv run --extra train python scripts/train.py \
  MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 20000 \
  --resume-checkpoint <STAGE1_CHECKPOINT>
```

### Play and Export

Inspect a PyTorch checkpoint in its training environment:

```bash
uv run --frozen --extra train adapt-play <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

Alternatively, run the Python script directly:

```bash
uv run --extra train python scripts/play.py <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

Export a checkpoint for deployment:

```bash
uv run --frozen --extra train adapt-export <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --output <OUTPUT_PATH>
```

Alternatively, run the Python script directly:

```bash
uv run --extra train python scripts/export_onnx.py \
  <TASK_NAME> \
  --checkpoint <CHECKPOINT_PATH> \
  --output <OUTPUT_PATH>
```

The export command creates a validated ONNX policy and its matching JSON
configuration. Released checkpoints and task names are listed in
[`logs/rsl_rl/README.md`](logs/rsl_rl/README.md).

### MuJoCo Deployment

Run one of the released policies in MuJoCo:

```bash
uv run --frozen adapt-sim --policy adapt --seconds 20
```

Alternatively, run the Python script directly:

```bash
uv run python deployment/deploy_mujoco_mjlab.py --policy adapt --seconds 20
```

Available released policies are `adapt`, `baseline`, and `lightstep`. The policy
name resolves the matching ONNX and JSON files from `logs/rsl_rl/`; a task name
is not required for deployment. To load a custom export, use
`--checkpoint <ONNX_PATH>` and keep its matching JSON file in the same directory.

`MujocoRobot` reads a robot-state snapshot and applies controls,
`ObservationGenerator` builds observations and their history, and
`PolicyManager` selects the active policy. The control loop in
[`deployment/deploy_mujoco_mjlab.py`](deployment/deploy_mujoco_mjlab.py) runs
observation → ONNX inference → action processing → MuJoCo step → MoMo update.
ADAPT uses the approximate MoMo calculation in
[`observer/momo.py`](observer/momo.py).

To load an ONNX/JSON pair directly without using `manifest.json`:

```bash
uv run --frozen adapt-sim --checkpoint <ONNX_PATH> --seconds 20
```

```bash
uv run python deployment/deploy_mujoco_mjlab.py \
  --checkpoint <ONNX_PATH> \
  --seconds 20
```

To use a wired Xbox-compatible controller in MuJoCo, first install the
`controller` extra. Add policies with `--add-policy`, then pass `--controller`:
the left stick sets forward/lateral velocity, the right stick sets yaw rate,
and releasing START/SELECT switches to the next/previous policy. Switching
keeps the robot pose but resets the selected policy's observation history and
disturbance observer.

```bash
uv run --frozen --extra controller adapt-sim --policy adapt --add-policy baseline --controller --seconds 20
```

```bash
uv run --extra controller python deployment/deploy_mujoco_mjlab.py \
  --policy adapt --add-policy baseline --controller --seconds 20
```

For a custom ONNX/JSON pair, use `--add-policy <NAME>=<ONNX_PATH>`. Runtime
switching is available in MuJoCo; the real-robot entry point remains
single-policy and requires explicit control enablement. Without `--controller`,
MuJoCo uses the fixed `--command` value.

### Real-Robot Deployment

Start with read-only state and LIO inspection:

```bash
uv run --frozen --extra robot adapt-real --net eno1 --lidar-frame raw --state-only
```

Alternatively, run the Python script directly:

```bash
uv run --extra robot python deployment/deploy_real_lio_mjlab.py \
  --net eno1 \
  --lidar-frame raw \
  --state-only
```

To load an ONNX/JSON pair directly without using `manifest.json`:

```bash
uv run --frozen --extra robot adapt-real \
  --net eno1 \
  --lidar-frame raw \
  --checkpoint <ONNX_PATH> \
  --state-only
```

```bash
uv run --extra robot python deployment/deploy_real_lio_mjlab.py \
  --net eno1 \
  --lidar-frame raw \
  --checkpoint <ONNX_PATH> \
  --state-only
```

Replace `eno1` with the robot network interface and select `raw` or
`compensated` according to the Livox driver configuration. Low-level control
must be enabled explicitly with `--enable-control` after hardware-specific
validation.

### Unitree G1 Code Configuration

The on-robot setup for the Unitree G1 requires
[FAST-LIO](https://github.com/hku-mars/FAST_LIO) to provide the base linear
velocity estimate used by the deployment code.

Detailed installation and configuration instructions for our G1 setup are
still in preparation.

## Citation

```bibtex
@misc{lyu2026adapt,
  title={ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion},
  author={Bofan Lyu and Jindou Jia and Kuangji Zuo and Yanshuo Lu and Shijia Han and Gen Li and Boyu Ma and Jingliang Li and Geng Li and Jianfei Yang},
  year={2026},
  eprint={2606.16542},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
