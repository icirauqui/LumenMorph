# Reusable tubular benchmark

The opt-in `backend.src.benchmark` producer builds a versioned synthetic suite for
global/local reconstruction, deformation recovery, static false-motion controls,
and moving versus stationary camera acquisition. It uses the retained generator's
tube geometry, texture, camera, and rendering conventions. Historical datasets and
the existing generator entry point are preserved.

`configs/tubular_benchmark_v1.json` is the design specification. A generated suite's
`SPECIFICATION.json`, provenance, and completed-stage hashes are authoritative for
that particular run. This document describes the schema and usage; it does not
assert that generation, reconstruction, or recovery qualification has passed.

## Design and independence

The initial specification contains six independently seeded scene families, each
with static, FEM-driven, and prescribed non-FEM motion: **18 version packages,
v1–v16**. Every package offers both moving and stationary action streams. Families
share their rigid reference images across conditions. Within a family, geometry,
texture, and camera paths are controlled across conditions, and moving/stationary
acquisitions use exactly the same material deformation states. These paired
variants are not independent scene samples.

Families f03 and f06 carry the stronger prescribed-kinematics regime: both FEM
and non-FEM basis peak amplitudes are twice those in the moderate families. They
provide development and final-test large-motion challenges. Amplitude regime is
coupled to family in this initial suite; differences between moderate and strong
families must not be attributed solely to amplitude. The stronger FE-derived
fields remain exact prescribed motion, not validated large-deformation mechanics.

| Families | Split | Purpose |
|---|---|---|
| f01–f03 | development | Repeated diagnosis, method development, regression |
| f04 | validation | Model/configuration choices under declared protocols |
| f05–f06 | final_test | Frozen evaluation after method choices are fixed |

Generation QA, checksums, signal/visibility audits, and numerical mechanics checks
are allowed for every split. Repeated inspection of inverse recovery outcomes
would consume final-test independence; creating another point split does not
restore it. Existing v1/v2 and C280 remain separate historical evidence.

The nominal camera is a calibrated pinhole camera. These are textured geometric
scenes, not clinical photometric simulations. None supplies tissue validation.

## Acquisition and temporal contrasts

Every family has a 90-frame undeformed reference scan and each action stream has
90 frames, sampled at 15 Hz. Stationary action keeps the terminal reference camera
pose exactly fixed. Moving action starts at a different trajectory position: it
is a separate acquisition after camera repositioning, not a continuous extension
of the reference trajectory. Local windows must not silently span that join.

The action schedule has initial rest (frames 0–14), a smooth rise, a 2% low-signal
plateau (30–44), another rise, a full-amplitude plateau (60–74), and return to rest
at frame 89. The three spatial bases use distinct transition timing but shared
plateau amplitudes. Static conditions use zero displacement at every frame.

The initial R0 geometry preflight used a 20% low plateau and moderate basis peaks
in every family. Its retained results motivated the explicit R1 design change to
2% and the two strong families before production generation or inverse recovery
evaluation. R0 remains historical design evidence; its signal measurements do not
describe R1. Scientific acceptance thresholds were not changed by this revision.

The declared nominal contrast compares 0–14 against 60–74; the low-signal contrast
compares 0–14 against 30–44. They are predefined contrasts, not selections based
on inverse-method performance. Full states support other questions, but changing
an evaluation contrast requires an explicit new protocol. Signal and visibility
QA must report actual supported material displacement, including inadequate
support; the presence of deformation elsewhere in the scene is not sufficient.

## Directory schema

```text
suite/
  SPECIFICATION.json, FROZEN.json
  provenance/                  source archive, hashes, revisions, environment
  families/f01/
    public/intrinsics.json, acquisition.json
    public/reference/          common rigid RGB scan and frame manifest
    evaluation/                geometry, texture, bases, mechanics, exact poses
    reference_truth/           exact reference states, dense maps, forward flow
  v1/                          one family/condition package
    DATASET.json
    public/
      intrinsics.json          link to common calibration
      acquisition.json         link to acquisition contract
      reference/               link to common rigid reference
      moving/rgb/, frames.json
      stationary/rgb/, frames.json
    evaluation/
      family/                  link to common private geometry/mechanics
      moving/
      stationary/
    reconstruction/            optional estimated outputs; never assumed present
```

For each stream, `frames.json` names PNG basenames relative to its `rgb/` folder.
Its intrinsics path is relative to the manifest's directory. Public timestamps
are shared acquisition time: reference begins at zero and action begins at 6 s.
Exact states also store action-local time separately. Package links are relative;
copy or archive the entire suite to preserve them, or explicitly dereference
links when exporting one package.

**Permitted inference input is `public/` and independently estimated
`reconstruction/` outputs.** Everything in `evaluation/`, family private files,
and `reference_truth/` is privileged evaluator information. Exact poses, texture
coordinates, material identity, depth, flow, deformation bases, and mechanics
must not be used as image-derived observations. A method using such inputs is a
distinct known-information control and must be labelled accordingly.

## Exact geometry and dense truth

Each private stream contains:

| Artifact | Meaning |
|---|---|
| `states.npz` | Exact float32 raster input vertices, float64 displacement from float32 reference, `R_wc`, camera centers, times |
| `dense/NNNNNN.npz` | Camera-z depth, triangle ID, physical material barycentrics, world face normals, valid mask |
| `flow_forward/NNNNNN.npz` | Pixel flow from frame N to N+1 plus validity and visibility categories |
| `CONVENTIONS.json` | Coordinates, units, material/state, clipping, and inference exclusion |
| `QUALITY.json` | Per-frame image features and geometric surface coverage |
| `COMPLETE.json` | Stage file checksums written after completion |

The RGB image and its dense truth are produced from **one perspective-correct
z-buffer**. Historical affine raster buffers remain available for historical
audits; they must not be substituted for this new truth API.

The benchmark opts into `high_precision=True`: exact stored float32 geometry and
camera values are promoted before subtraction or matrix multiplication; camera
transforms, projected vertices, raster weights, z-buffer comparisons, and flow
visibility use float64 arithmetic. Exported depth and material barycentrics remain
float32. This avoids amplified projection-rounding error on grazing triangles
without changing the stored physical inputs or loosening quality thresholds.
The generic APIs retain their original arithmetic by default. `CONVENTIONS.json`
records the mode, and the evaluator reader propagates it when deriving flow.

Tube RGB explicitly enables periodic angular U interpolation: seam-crossing
triangles with U range greater than 0.5 have their low-U vertices unwrapped by
one before material interpolation, followed by U modulo one. This prevents
welded angular seam triangles from sampling the middle of the texture. It changes
RGB at the seam without changing geometry, material barycentrics, depth, normals,
or flow. The generic API defaults to the historical nonperiodic convention.
Nearest-texel indexing retains `floor(U*(texture_width-1))`, preserving nonseam
RGB compatibility rather than resampling the whole texture with a new convention.
Closed terminal rings use fan triangles over the existing ring vertices. Their
inherited UVs define an arbitrary closure texture, not a canonical planar mapping
or a physical/clinical surface model; interpret visible end-cap texture accordingly.

Coordinates follow `p_camera = R_wc.T @ (p_world - C_world)`: camera x right, y up,
z forward; `u=fx*x/z+cx`, `v=cy-fy*y/z`. Pixel centers are `(column+.5,row+.5)`.
Depth is camera z, not ray distance. Length units are abstract scene units; they
are not automatically millimetres. Normals are unit world face normals following
triangle winding, without camera-facing reorientation. Quaternion or COLMAP
conversion must respect the explicit coordinate reflection in `CONVENTIONS.json`.
The exact stored float32 `R_wc` is not silently orthogonalized: physical ray audits
invert its declared transpose transform, rather than assuming exact orthogonality.

`triangle_id` plus `material_barycentric` identifies a persistent material point.
The same triangle vertices and weights evaluate that point at every frame.
Barycentrics are physical surface weights, not affine image-space weights.
Background has `valid=false`, triangle `-1`, and NaN depth/weights/normals. Always
mask it before numeric use.

The renderer discards an entire triangle if any camera-space vertex has z <=
1e-5; it does not clip polygons against the near plane. Independent ray QA must
detect any consequential coverage holes. This limitation is part of the recorded
renderer contract, not an assumption that every scene is safe from clipping.

## Flow and visibility

Forward flow is destination minus source in `(u,v)` pixels after advecting the
same material point. The last frame has no stored next-frame flow by definition.
It is not a missing asset. Flow remains defined for occluded or outside-image
points with a finite positive-depth projection.

`flow_valid` describes finite positive-depth projection, **not** visibility.
For a source-valid pixel, the following categories are mutually exclusive:

- `visible`: inside target image and consistent with its front surface;
- `occluded`: an actual target surface lies closer on the same ray;
- `out_of_frame`: target projection lies outside image bounds;
- `behind_camera`: no finite positive-depth projection;
- `unresolved`: inside-image projection inconsistent with rendered target
  geometry, including possible whole-triangle near-plane discard.

Visibility uses a continuous target z-buffer evaluated at actual advected
subpixel endpoints. It does not round to the nearest target depth pixel.
Default depth tolerance is `1e-5 + 1e-5*abs(z)` in world units. Boolean masks are
false at source background pixels; their union equals the source-valid mask.

Backward and long-range flow are exactly derivable from the stored topology,
material weights, and states. They need not duplicate large files on disk.

## Mechanics scope

The new mechanics path assembles linear isotropic C3D6 elements **after mapping
the reference shell onto the curved tube**. Both end rings are fixed, and smooth
surface patch loads create basis displacements. Saved data include FE nodes,
elements, material parameters, constraints, load/displacement/reaction bases,
strain/stress/energy, and the assembled stiffness matrix. Rendered surface fields
are bilinear interpolants of outer FE nodes, with periodic angular interpolation.

This is prescribed kinematic truth generated from a documented linear model.
Small strain diagnostics must be checked before interpreting it as a faithful
linear mechanics example, and no claim of tissue realism follows. The non-FEM
condition uses independent prescribed vector Gaussian fields with smooth fixed
ends; it carries no equilibrium or constitutive-model claim. “FEM” does not mean
every downstream estimator uses exactly the same mesh, boundaries, or basis.

## Reader examples

From the generator checkout with its Python environment:

```python
from backend.src.benchmark_reader import PublicDataset, EvaluationDataset

inputs = PublicDataset('/absolute/path/to/suite/v2')
paths = inputs.public_paths('stationary')
image_bgr = inputs.rgb(0, 'stationary')
frames = inputs.frames('stationary')

# Evaluator only: never pass these objects to inference.
truth = EvaluationDataset('/absolute/path/to/suite/v2')
dense = truth.dense(60, 'stationary')
valid = dense['valid']
ids = dense['triangle_id'][valid]
weights = dense['material_barycentric'][valid]
xyz = truth.material_points(ids, weights, frame=60)
displacement = truth.material_displacement(ids, weights, frame=60, from_frame=0)
backward = truth.compute_flow(source_frame=61, target_frame=60)
```

Reader queries write nothing. They reject invalid triangle IDs, malformed
barycentrics, and out-of-range frame numbers. Material queries accept arbitrary
valid triangle/weight arrays, supporting sparse evaluation without a separate
custom dataset export. `PublicDataset` never opens ground-truth files.

## Reproduction and stage discipline

```bash
backend/.venv/bin/python -m backend.src.benchmark freeze --root /new/suite --spec configs/tubular_benchmark_v1.json
backend/.venv/bin/python -m backend.src.benchmark prepare --root /new/suite --family f01
backend/.venv/bin/python -m backend.src.benchmark render --root /new/suite --family f01 --stream reference
backend/.venv/bin/python -m backend.src.benchmark package --root /new/suite --family f01 --condition fem
backend/.venv/bin/python -m backend.src.benchmark render --root /new/suite --family f01 --condition fem --stream stationary
backend/.venv/bin/python -m backend.src.benchmark render --root /new/suite --family f01 --condition fem --stream moving
```

Each stage requires fresh output directories and refuses replacement. Failed or
interrupted stages are not complete merely because some files exist. Preserve
failed production attempts with their diagnosis; use a fresh versioned stage
for corrected generation. Before non-freeze CLI stages, the producer checks the
frozen source mapping, preventing unrecorded live source changes. Keep the exact
environment and external solver revision/source state from provenance.

Generation, independent render/mechanics QA, estimated reconstruction, and inverse
recovery validation are distinct milestones. A complete synthetic dataset supplies
quantitative truth; it does not guarantee identifiable recovery from images,
successful COLMAP reconstruction, or a positive FEM result. Failed coverage,
low-signal cases, and adverse method outcomes must remain in the research record.

## Portable source replay and mechanics reader (2026-09-11)

New freezes use canonical logical archive paths for the generator and the actual
solver selected by `CONTINUUM_FEM_SRC` (or the sibling default). Recorded absolute
roots describe provenance, but verification resolves against the current source
locations. Archives without Git metadata record unavailable revision/status;
the actual source hashes still identify the code. Installation metadata is now
included in new archives. Original frozen suites keep their existing source
verification contract and bytes. New source changes do not retroactively become
the producer of the existing bank.

`EvaluationDataset.mechanics(frame, stream)` derives applied coarse-FE node
motion, loads, reactions, Gauss strains/stresses, and total strain energy from
the retained bases. It returns zero for static action/reference acquisitions
and rejects non-FEM actions, whose stored family FE bases were not applied.
Gauss arrays preserve the producer's element-major six-point C3D6 order;
engineering strain order is xx, yy, zz, xy, yz, zx (shears doubled), whereas
stress shears are tensor components. Total energy is `0.5*u.T*K*u` including
cross terms. Adding saved basis energies is incorrect for superposed motion.
Time-composed element energies are not returned by this API.

```python
mechanical_state = truth.mechanics(60, 'stationary')  # FEM package only
energy = mechanical_state['strain_energy']
```

These are coarse generating-FE fields, not spatial derivatives of the separately
interpolated render mesh. They carry abstract consistent units; the configuration
key `young_modulus_pa` alone does not establish an SI calibration. No measured
tissue parameters or constitutive/clinical validation follow from this export.


## Pinned numerical installation

`backend/requirements-benchmark.txt` pins the three packages required for
benchmark generation, readers, numerical QA and the flow example. A fresh
installation of NumPy 2.4.4, SciPy 1.17.1 and OpenCV-Python 4.13.0.92, in an
isolated CPython 3.12.13 venv on Linux x86_64, reproduced the complete f01
preparation (35 arrays), texture and first reference RGB/dense frame from
relocated source archives without Git metadata. The Python executable and host
were retained, but no packages were inherited from the original environment.
This verifies a bounded fresh-package replay, not a full-bank rerender,
COLMAP rebuild or cross-platform bitwise reproducibility. See the root README
for executable installation commands. Solver source remains separately required.
