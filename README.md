# LumenMorph

**A Reproducible Synthetic Benchmark for Deformable Endoscopic Reconstruction**

LumenMorph is a deterministic generator and reference implementation for
synthetic deformable tubular endoscopic sequences. It produces controlled image
observations together with dense geometric, kinematic and mechanical
annotations, so reconstruction and deformation-analysis methods can be tested
against known conditions.

Its public Paper 0 resource is designed for deformable 3D reconstruction,
global/local reconstruction workflows, correspondence and optical-flow
evaluation, and controlled appearance-robustness studies. It is not a clinical
simulator or a claim of validated tissue mechanics.

Source code: <https://github.com/icirauqui/LumenMorph>.

## Documentation

- [Install LumenMorph](docs/INSTALLATION.md) for the supported Python and browser setup.
- [Use the browser workspace](docs/tube_user_guide.md) to design and generate a tube sequence.
- [Understand the system architecture](docs/ARCHITECTURE.md), for the supported React UI.
- [Work reproducibly](docs/REPRODUCIBILITY.md) before generating or evaluating research data.
- [Use the fixed tubular benchmark](docs/TUBULAR_BENCHMARK.md) and its [evaluator](docs/EVALUATION.md).
- [Contribute](CONTRIBUTING.md), [cite](CITATION.cff), or [report a security issue](SECURITY.md).

## What It Generates

- Plane scene sequences with deforming terrain-like surfaces.
- Tube scene sequences (camera inside tube) with bumps, curvature, and deformation.
- RGB frames and preview video (`preview.mp4`) plus camera and dense-truth
  metadata.

## Features

- Deterministic generation with fixed seeds.
- Procedural texture generation with ORB feature quality checks.
- CPU triangle rasterizer with z-buffer and UV mapping.
- Photometric noise and brightness jitter.
- Progress bars during generation (`tqdm`).
- Dataset validation and sequence preview utilities.

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- Node.js 22 LTS or newer and npm, for the supported browser workspace.
- A recent desktop browser. The application is designed for local use.

Install the locked backend environment and the browser workspace from the
repository root:

```bash
uv sync --project backend --locked
npm --prefix frontend ci
```

The `backend/uv.lock` file is the supported full-environment lock. The
`requirements*.txt` files remain narrower component-specific environments for
benchmark or rendering work; they are not substitutes for the locked app setup.
See [the installation guide](docs/INSTALLATION.md) for platform notes and
troubleshooting.

## Fixed tubular research benchmark

For scoring supplied depth, optical-flow, normal or material-displacement
predictions, use [the evaluation guide](docs/EVALUATION.md). The evaluator reports
fixed-population coverage and errors without fitting a transform to truth.

The current research bank is produced by `backend.src.benchmark`, a separate
staged producer from the generic plane/tube CLI below. See
[the benchmark schema and reproduction guide](docs/TUBULAR_BENCHMARK.md) for
shared references, moving/stationary actions, dense material truth, mechanics,
and evaluator access. Existing data should be reused for registered experiments;
a new suite is appropriate only for a documented coverage gap.

For numerical benchmark generation, reading, QA and the flow example, use the
smaller pinned runtime rather than the full API environment:

```bash
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python -r backend/requirements-benchmark.txt
.venv-benchmark/bin/python -m backend.src.benchmark --help
```

The three numerical packages were freshly installed into an isolated venv and
verified with CPython 3.12.13 on Linux x86_64. A Git-free relocated source replay
reproduced f01 preparation and its first reference RGB/dense frame exactly. This
is not a guarantee of identical bytes across other Python versions/platforms.
The small runtime does not include the generic preview CLI, API/browser,
plotting or test dependencies; the full requirements above serve those uses.

The FEM producer uses the pinned `ContinuumFEM` submodule, which exposes the
`continuum_fem` package. Clone with `--recurse-submodules` (as shown above), or
run `git submodule update --init --depth 1` after cloning. For another location,
set `CONTINUUM_FEM_SRC=/absolute/path/to/continuum-fem/src` before freezing and
producing a suite. Freeze records that actual source location in the portable
source archive. A source archive without Git metadata is supported. The reader
does not require the solver to access an existing bank.

```bash
backend/.venv/bin/python -m backend.src.benchmark --help
```

LumenMorph and its accompanying documentation are licensed under
[MIT](LICENSE), as approved by the copyright holder on 2026-09-11.
The separately maintained `ContinuumFEM` submodule is also MIT-licensed.
Other dependencies retain their own licenses. Generated Paper 0 benchmark data
are separately licensed under CC BY 4.0; distribute the benchmark's
accompanying data-license notice and preserve third-party source/binary notices.

## Quick Start

Generate one quick plane sequence:

```bash
backend/.venv/bin/python -m backend.src.cli generate \
  --config configs/default.json \
  --out /tmp/quick_plane \
  --single-sequence \
  --override-num-frames 120
```

Generate one quick regular tube sequence:

```bash
backend/.venv/bin/python -m backend.src.cli generate \
  --config configs/tube_default.json \
  --out /tmp/quick_tube \
  --single-sequence \
  --override-num-frames 120
```

Generate one quick hard tube sequence:

```bash
backend/.venv/bin/python -m backend.src.cli generate \
  --config configs/tube_hard_endoscope.json \
  --out /tmp/quick_tube_hard \
  --single-sequence \
  --override-num-frames 120
```

Generate one quick tube sequence with localized side-region deformation:

```bash
backend/.venv/bin/python -m backend.src.cli generate \
  --config configs/tube_localized.json \
  --out /tmp/quick_tube_localized \
  --single-sequence \
  --override-num-frames 120
```

Preview and validate:

```bash
backend/.venv/bin/python -m backend.src.cli preview --sequence /tmp/quick_plane/train/seq_000
backend/.venv/bin/python -m backend.src.cli validate --dataset /tmp/quick_plane
```

Run the localhost API service used by the React control surface:

```bash
backend/.venv/bin/python -m backend.src.cli api --host 127.0.0.1 --port 8765
```

Then in another terminal run the supported browser workspace:

```bash
cd frontend
npm run dev -- --host localhost --port 5173
```

Open `http://localhost:5173`.

Generate all bundled full datasets (plane + regular tube + hard tube):

```bash
mkdir -p data/full && \
for cfg in configs/default.json configs/tube_default.json configs/tube_hard_endoscope.json; do
  name="$(basename "${cfg%.json}")"
  backend/.venv/bin/python -m backend.src.cli generate --config "$cfg" --out "data/full/$name"
done
```

Optional validation for all generated datasets:

```bash
for d in data/full/*; do
  backend/.venv/bin/python -m backend.src.cli validate --dataset "$d"
done
```

## CLI Reference

### `generate`

```bash
backend/.venv/bin/python -m backend.src.cli generate --config <path> --out <dir> [options]
```

Arguments:

- `--config` (required): Path to JSON config file.
- `--out` (required): Output dataset directory.
- `--max-sequences` (optional): Limit total generated sequences from config order.
- `--single-sequence` (optional): Force one sequence only.
- `--override-num-frames` (optional): Override `sequence.num_frames` for this run.
- `--override-fps` (optional): Override `sequence.fps` for this run.
- `--no-progress` (optional): Disable progress bars.

Notes:

- `--single-sequence` takes precedence over multi-sequence runs.
- By default, generation follows `sequence_count` + train/val/test split in config.

### `validate`

```bash
backend/.venv/bin/python -m backend.src.cli validate --dataset <dir>
```

Arguments:

- `--dataset` (required): Dataset root directory containing `dataset_manifest.json`.

Checks include:

- Required files/folders and counts.
- CSV schema and numeric sanity.
- Camera motion and clearance checks.
- Mean, fifth-percentile, and minimum sampled-frame ORB feature statistics; `validation.min_orb_features_frame` rejects featureless tails.
- MP4 readability.

### `preview`

```bash
backend/.venv/bin/python -m backend.src.cli preview --sequence <dir>
```

Arguments:

- `--sequence` (required): Sequence directory path, e.g. `.../train/seq_000`.

Returns quick stats:

- Frame count and resolution.
- Mean ORB features.
- Preview MP4 existence/readability.

### `api`

```bash
backend/.venv/bin/python -m backend.src.cli api [--host 127.0.0.1] [--port 8765]
```

Runs the local FastAPI backend used by the LumenMorph control surface.
Main routes:

- `GET /v1/health`
- `GET /v1/schema`
- `GET /v1/presets`
- `POST /v1/config/normalize`
- `POST /v1/preview/jobs`
- `POST /v1/generate/jobs`
- `GET /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/cancel`
- `GET /v1/jobs/{job_id}/frames/{index}`
- `GET /v1/jobs/{job_id}/result`

## Available Config Profiles

- `configs/default.json`: Plane baseline.
- `configs/tube_default.json`: Regular tube profile.
- `configs/tube_hard_endoscope.json`: Hard tube profile (tighter/curvier/harder).
- `configs/tube_localized.json`: Tube profile with localized side-region deformation.

## Config Structure (Top-Level)

Main keys:

- `version`, `seed`, `scene_type`
- `sequence_count`, `default_preset_cycle`
- `intrinsics`, `sequence`, `camera`
- `texture`, `photometric`, `output`, `validation`
- `difficulty_presets`
- `plane` (plane features)
- `tube` (tube features, required for `scene_type: "tube"`)

## Plane-Specific Controls (`plane`)

- `landmark_count`: number of static non-deforming landmarks.
- `landmark_height_min`, `landmark_height_max`: landmark height range.
- `landmark_sigma_min`, `landmark_sigma_max`: landmark footprint range.
- `rigid_ratio`: approximate rigid fraction of scene (0.0 to 1.0).
- `rigid_include_landmarks`: include landmarks in rigid mask.
- `rigid_edge_softness`: smoothing passes on rigid boundaries.
- `rigid_colorize`: enable rigid-area tint overlay.
- `rigid_tint_bgr`: tint color in OpenCV BGR order.
- `rigid_tint_strength`: tint blend strength.
- `flip_vertical_view`: flips image vertically for ground-flying view.

Rigid-area seeding patterns are sampled from corner, side, or middle-band styles.

## Tube-Specific Controls (`tube`)

- `length`, `radial_samples`, `longitudinal_samples`
- `base_radius`, `min_radius`
- `bump_count`, `bump_amp_min`, `bump_amp_max`
- `bump_sigma_long`, `bump_sigma_theta`
- `curve_amplitude_x`, `curve_amplitude_z`
- `curve_frequency_x`, `curve_frequency_z`
- `camera_offset_ratio`, `camera_wobble_ratio`, `camera_wobble_hz`
- `lookahead_distance`
- `localized_deform_count`: number of localized deformation regions (0 disables localized motion).
- `localized_longitudinal_min`, `localized_longitudinal_max`: normalized tube-length range where local regions can appear.
- `localized_theta_center`, `localized_theta_jitter`: side bias in normalized angular coordinate `[0,1)`.
- `localized_sigma_long_min/max`, `localized_sigma_theta_min/max`: local region footprint ranges.
- `localized_amp_scale`, `localized_negative_ratio`: scale/sign mix for localized deformation amplitude.
- `localized_tint_enabled`, `localized_tint_bgr`, `localized_tint_strength`: optional tint overlay for localized tube regions.
- `deformation_flow_mode`: `linear` (one-way) or `cyclic` (wobbling loop) for localized area movement.
- `deformation_flow_cycles`: number of oscillation cycles used in cyclic mode.

## Output Layout

Dataset root:

- `dataset_manifest.json`
- `config_used.json`
- `train/`, `val/`, `test/`

Per sequence (`train/seq_000/`):

- `frames/000000.png ...`
- `preview.mp4`
- `intrinsics.json`
- `poses.csv`
- `deformation_params.csv`
- `mesh_static.npz`
- `mesh_z/*.npy`
- `sequence_meta.json`

## Reproducibility

- Same config + seed yields deterministic geometry/deformation and metadata.
- Config hash is stored in `dataset_manifest.json`.
- Generated images may differ across operating systems or dependency versions;
  retain the lockfile, configuration, seed and generator commit for each run.
- Exact generated geometry and kinematics are synthetic truth. They do not make
  the chosen tissue parameters clinically validated.

## Testing

Run tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 backend/.venv/bin/python -m pytest -q
```
