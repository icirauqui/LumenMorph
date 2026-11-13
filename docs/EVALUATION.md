# Evaluate predictions on the tubular collection

`backend.src.benchmark_evaluate` scores a supplied prediction file. It runs no
estimator, chooses no algorithm settings and changes no dataset files. It
supports the raster collection. Apply the suite's family split and exposure rules to every task.

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

These `benchmark/v2` commands require the complete f01 FE package, not the
compact f07/f08 example. For the latter, follow [first use](FIRST_USE.md) and
`examples/consume_collection.py`. See the [collection guide](COLLECTION_GUIDE.md)
for operational visibility classes, reconstruction components and pose diagnostics.

```sh
backend/.venv/bin/python -m backend.src.benchmark_evaluate \
  --package "$SDTUBE_DATA/benchmark/v2" \
  --task flow --frame 52 --stream moving \
  --predictions /tmp/my_flow.npz \
  --output ./my_run/flow_52.json

backend/.venv/bin/python -m backend.src.benchmark_evaluate \
  --package "$SDTUBE_DATA/benchmark/v2" \
  --task displacement --frame 0 --target 67 --stream stationary \
  --coordinate-frame benchmark-world \
  --predictions /tmp/my_displacement.npz \
  --output ./my_run/displacement_0_67.json
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

## Appearance scoring

`backend.src.appearance_evaluate` accepts the same `prediction`/optional `mask`
NPZ format. Its tasks are `depth`, `normals`, `displacement` and
`projected_flow`. Depth and displacement errors are in metres; normals use the
appearance-world axes and oriented angular error. Temporal tasks require
`--target`. Normals and displacement require
`--coordinate-frame appearance-world-meters`.

```bash
.venv/bin/python -m backend.src.appearance_evaluate \
  --family "$SDTUBE_DATA/appearance/families/f01" \
  --task depth --frame 0 --predictions ./depth.npz --output ./depth_score.json
```

The primary depth/normal/displacement population is every valid source pixel.
Projected flow additionally requires positive target depth and an in-image
projection. **No target-occlusion classification is supplied or inferred** by
this scorer. All methods share the same eligibility rule; missing predictions
reduce coverage rather than remove cases. The five path-traced appearance
conditions share states internally. Comparing them with raster images also
involves the documented coordinate/rounding conversion and rendering pipeline.

## Globally aligned monocular material tracks

Three evaluation contracts must be distinguished: pixel flow has no 3D scale
ambiguity; predictions supplied in known benchmark axes/scale use the dense
scorer directly; free monocular reconstruction needs a declared global gauge.
`backend.src.monocular_evaluate` implements the third contract for one concrete
format. It evaluates existing predictions; it does not run SfM or estimate
material tracks from ground truth.

The prediction NPZ contains:

| Key | Shape | Meaning |
|---|---|---|
| `frame_ids` | T integer | Unique requested acquisition frame IDs |
| `centers` | T×3 | Predicted camera centers in one common prediction world |
| `rotations` | T×3×3 | Camera-to-prediction-world rotations; camera x right, y up, z forward |
| `reference_pixels` | N×2 integer | Unique source-image `(row,column)` observations defining tracks |
| `points` | T×N×3 | Predicted positions of those same material tracks |
| `mask` | T×N boolean, optional | Available predictions, never the evaluator population |
| `normals` | T×N×3, optional | Oriented normals in prediction-world axes |

The protocol JSON declares `schema: material_tracks_alignment_v1`, a
`component_id`, `reference_frame`, the complete requested `frame_ids`, `anchor_frame_ids`, `reference_pixels_sha256` and optionally
`minimum_second_singular_ratio` (default 1e-6). Choose frames, pixels, anchors,
algorithm and protocol before inference or inspecting truth. Hash the ordered
reference `(row,column)` array as C-contiguous little-endian int64 bytes with
SHA-256; its identity is checked against the protocol. Keep every requested
frame in the prediction, using nonfinite values and false masks for missing
outputs; omitting difficult frames is rejected. Freeze both files
before evaluation; the evaluator hashes and loads them before opening truth.

A public-input helper writes the complete protocol and the ordered observation grid:

```bash
PYTHONPATH=. .venv/bin/python examples/prepare_monocular_protocol.py \
  --package "$SDTUBE_DATA/benchmark/v20" --stream moving \
  --anchors 0 80 160 239 --output ./protocol_inputs
```

For example, f07's predeclared action anchors `[0, 80, 160, 239]` sample its
trajectory. A component with absent anchors, fewer than three noncollinear
anchors, nonfinite anchor centers or insufficient second/first centered singular
value ratio fails alignment. It receives a saved failure with zero scored
coverage; no frame-specific fallback is attempted. Anchor requirements apply
to both prediction and reference camera centers.

```bash
.venv/bin/python -m backend.src.monocular_evaluate \
  --package "$SDTUBE_DATA/benchmark/v20" --stream moving \
  --predictions ./tracks.npz --protocol ./protocol.json --output ./aligned_score.json
```

One least-squares proper positive-scale similarity `s R X + t` aligns the
predeclared camera anchors. The same transform applies to every point and
camera center; camera orientations become `R R_camera`, displacements become
`s R u`, and normals are rotated and normalized. The transform is never fitted
per frame. Each independently reconstructed component requires its own declared
submission; separate aligned components are not a joint reconstruction.

The score reports the reference-image sampling fraction separately from track
coverage. A sparse predeclared cohort does not measure complete surface support.

Outputs explicitly say **ground-truth-aligned** and do not claim recovered
absolute scale. Camera position/rotation and persistent material position,
displacement and optional normal error retain coverage and conditional versus
full-population scores. Every valid reference material observation remains in
the requested population across target frames, including occlusions. Reference
background is ineligible. Failed or nonfinite predictions remain missing.

### Material-coordinate decoding and materialized references

Read each raster acquisition's `evaluation/<stream>/CONVENTIONS.json`.
`material_weight_policy: unit_sum` means that stored weights are promoted to
float64 and divided by their sum before geometric interpolation. This restores
partition of unity after float32 storage and preserves translation invariance.
The policy applies to material positions, displacement and projected flow; it
is explicit in the f07 configuration. Absence of the key, or `stored`, retains
historical arithmetic. Texture interpolation has its separate recorded sampling
convention; this policy does not recolor saved images.

`EvaluationDataset.material_weights()` implements that decoding, and geometric
query/evaluation methods use it automatically. A standalone consumer must apply
the same policy before combining world vertices. The public materialized layout
stores shared reference truth at `evaluation/reference/`; the reader also
recognizes the historical linked-family layout. Use `image_rgb()` for RGB in
both raster and path-traced data.

An independent full-pixel geometric census can be repeated using
[`tools/validation/README.md`](../tools/validation/README.md). Recorded producer
capsules and validation inputs identify the precise numerical scope separately
from arbitrary image-based reconstruction performance.
