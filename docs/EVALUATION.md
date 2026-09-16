# Evaluate predictions on the fixed tubular bank

`backend.src.benchmark_evaluate` scores a supplied prediction file. It runs no
estimator, chooses no algorithm settings and changes no dataset files. It
supports the complete existing bank without creating additional rendered
versions. Apply the suite's family split and exposure rules to every task.

The prediction file is an NPZ containing a real numeric array named `prediction`
and optionally a boolean H×W array named `mask`. No other keys are accepted.
The mask means “my method produced output here.” It **cannot select the scored
population**: every eligible pixel remains in the denominator even when the
mask is false. Resolution is the original image resolution; the evaluator does
not resize, fit scale, align coordinates or interpolate a prediction.

| Task | Prediction shape and convention | Fixed primary population | Error |
|---|---|---|---|
| `depth` | H×W positive camera-z depth in abstract scene length | Every source-valid surface pixel | Absolute depth error in scene length |
| `flow` | H×W×2 next-frame displacement `(du,dv)` in pixels | Source-valid, finite-positive-depth-projecting, target-visible pixels | Euclidean endpoint error in pixels |
| `normals` | H×W×3 oriented world-space surface normal | Every source-valid surface pixel | Angle in degrees, 0–180; magnitude normalized |
| `displacement` | H×W×3 material displacement from source to target in benchmark-world coordinates and scene length | Every source-valid surface pixel | Euclidean endpoint error in scene length |

World normals follow triangle winding; sign is meaningful, so opposite normals
have 180° error. Zero-length normals and nonpositive depths are invalid
predictions. Nonfinite values are missing predictions. Background is excluded
by the dataset validity mask, not by whether a method succeeds.

Every result reports the image/source denominator, eligible count, available
count, finite count, valid prediction count, missing/invalid count and coverage.
It gives explicitly **conditional** mean/median/p95 errors over valid predictions.
The corresponding **full-population** errors are null unless every eligible
pixel has a valid prediction. Empty support has null scores and is not complete
prediction support. Do not rank a partial-coverage conditional error against a
full-coverage result without disclosing the different support.

Flow additionally reports occluded and out-of-frame strata, with the same
coverage discipline, and all recorded visibility-category counts. The last
frame has no next-frame target, so its flow request is rejected. Displacement
may use any other stored frame in the same stream, including the same frame as
a zero-displacement control; it is not a continuous-time interpolator. Actual
static truth is zero and is scored without dividing by its magnitude.

## Prepare an estimator output

Your method consumes only its declared public images, calibration and permitted
estimated reconstruction inputs. Save its native-resolution dense prediction:

```python
import numpy as np

# prediction comes from your estimator, indexed on the source image pixels.
np.savez_compressed('predictions.npz', prediction=prediction, mask=available)
```

For displacement, a pixel refers to the material point visible in the **source
frame**. Predict its 3D motion to the target frame. The evaluator internally
uses privileged triangle IDs and physical barycentrics to locate that point;
these IDs are not an inference input. Target visibility is not required for
the primary displacement population. Missing/occluded-target predictions still
count as missing support. A method using known material IDs, depth or camera
poses is a separately declared known-information experiment.

For normals/displacement, `--coordinate-frame benchmark-world` is an explicit
acknowledgment that submitted values use the benchmark axes (and scene units
for displacement). It is not an estimated gauge conversion. An unconstrained
SfM output normally has an arbitrary similarity gauge; a protocol must define
how its gauge is established before submission. The evaluator never fits a
transform to truth. Label externally truth-aligned results as such; they are
not automatically absolute-world or scale recovery. Depth likewise assumes
the declared scene units and receives no truth-based rescaling.

## Run from the generator checkout

```sh
backend/.venv/bin/python -m backend.src.benchmark_evaluate \
  --package ../../datasets/generated/tubular_benchmark_v1/v2 \
  --task flow --frame 52 --stream moving \
  --predictions /tmp/my_flow.npz \
  --output ../../experiments/artifacts/my_run/flow_52.json

backend/.venv/bin/python -m backend.src.benchmark_evaluate \
  --package ../../datasets/generated/tubular_benchmark_v1/v2 \
  --task displacement --frame 0 --target 67 --stream stationary \
  --coordinate-frame benchmark-world \
  --predictions /tmp/my_displacement.npz \
  --output ../../experiments/artifacts/my_run/displacement_0_67.json
```

Use `--task depth` without a target, or `--task normals` with the world-frame
acknowledgment. `--stream reference` selects the rigid reference. Outputs must
be outside the immutable suite and existing result files are never overwritten.
Only compact JSON scores/hashes are written; no dense prediction or truth copy
is retained by the evaluator.

Results record prediction, evaluator, reader, relevant truth-file and suite-seal
identities. Freeze your method/configuration, frame pairs, population, split
use and desired primary metrics before looking at scores. This tool evaluates
one declared frame/pair; it does not turn frames or pixels into independent
scene samples. Aggregate with declared family weighting and retain every
requested frame's completion/coverage outcome. The fixed development flow
usage study in `benchmark_flow.py` supplies a separate reproducible example.

The evaluator's truth boundary is organizational, not access control: a user
with filesystem access can read evaluator assets. Good benchmark use depends
on declaring privileged information honestly and preserving split exposure.
Finite scores validate a submitted array against this synthetic task; they do
not certify clinical performance or tissue-model validity.
