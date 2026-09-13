# Released checkpoints

The selected models are stored under `logs/rsl_rl/4096*1/` with their original
filenames. The light-step directory is named `momo_softland`; its weights and
NoActorFeedback configuration are unchanged. Each `model_*.pt` has a matching `model_*.onnx` and
`model_*.json`. The `*` is a literal directory-name character; quote explicit paths
in shell commands.
ONNX includes the actor normalizer. JSON describes the matching observation order,
history lengths, PD gains, joint order, action scale, physical residual scale,
observer settings and the ONNX SHA-256. The loader rejects mismatched ONNX/JSON pairs.
[`manifest.json`](manifest.json) records release/source paths, sizes, hashes and iterations.

| Policy name | Training task | Actor input |
| --- | --- | --- |
| `baseline_stage1` | `Ref-NoMoMo-Flat-Unitree-G1` | 379 |
| `baseline` | `Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix` | 379 |
| `adapt_stage1` | `MOMO-Lumped-Flat-Unitree-G1-EffortScale` | 554 |
| `adapt` | `MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale` | 554 |
| `lightstep` | `MOMO-Lumped-Flat-Unitree-G1-LegEffortSoftStep-NoActorFeedback` | 939 |

All outputs are `[1,29]` raw actions. Deployment computes
`joint_target = action * action_scale + default_position` at 50 Hz, with physics
at 200 Hz. History is **term-major, oldest to newest**. The first observation
backfills each term's history using the current state.

Baseline has no observer input. ADAPT adds 35 effort-scaled residuals with five
history frames. Light-step retains 40 frames of a 14-dimensional leg channel,
but its **actor channel is zero**; its critic and training reward still use the observer.
The 939-dimensional model is not interchangeable with the 554-dimensional ADAPT model.

Stage 1 `.pt` files support Stage 2 continuation. Final `.pt` files support play,
further training and export. They include optimizer and curriculum state. Load
only trusted PyTorch checkpoints: the training library deserializes Python objects.
These model files are copied unchanged from the research codebase; historical
experiment recordings and paper plotting scripts are intentionally not included.

## Released model paths

Paths below are relative to this directory. The manifest's `path` field is the
released model path without its extension; `source` preserves the original research
path for attribution only. `files` records the actual filenames. `--policy` uses
the short policy names above to resolve `path`, never `source`.

| Policy name | PT / ONNX / JSON stem |
| --- | --- |
| `baseline_stage1` | `ref_g1/2026-05-13_02-11-50_stage1/model_14999` |
| `baseline` | `ref_g1/2026-05-14_02-15-28_stage2/model_34998` |
| `adapt_stage1` | `momo_g1_effortscale/2026-05-13_02-12-41_stage1/model_14999` |
| `adapt` | `momo_g1_effortscale/2026-05-14-_02-15-42_stage2/model_34998` |
| `lightstep` | `momo_softland/model_14999` |
