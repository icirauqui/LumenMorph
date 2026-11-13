# LumenMorph

**A Reproducible Synthetic Benchmark for Deformable Endoscopic Reconstruction**

LumenMorph is a deterministic generator and reference implementation for
synthetic deformable tubular endoscopic sequences. It produces controlled image
observations together with dense geometric, kinematic and mechanical
annotations, so reconstruction and deformation-analysis methods can be tested
against known conditions.

The Synthetic Deformable Tubular Benchmark is designed for deformable 3D reconstruction,
global/local reconstruction workflows, correspondence and optical-flow
evaluation, and controlled appearance-robustness studies. It is not a clinical
simulator or a claim of validated tissue mechanics.

Source code: <https://github.com/icirauqui/LumenMorph>.

Software version **2.0.3** provides generation, RGB readers, material queries,
evaluation and independent validation for the collection of eight seeded families.
The resource includes paired cameras and motion conditions, paired procedural
and file textures on f07/f08, and five path-traced appearance conditions on four families.

## First use: read, query and score the compact example

Use Python **3.12** with `venv`/`pip`, Git, curl, tar and SHA-256 tools on Linux.
Allow **2 GB free disk and 2 GB RAM**, internet access and a few minutes; no GPU,
Hub account, token or complete dataset is needed. The example download is 61.13 MB.
It contains three FE moving-camera frames from each of f07 and f08, with both
procedural and organic (`tissue`) RGB textures.

For a clean Docker shell, install Docker on the host and run:

```bash
docker run --name tubular-first-use --platform linux/amd64 -it \
  python@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e bash
```

This is official `python:3.12.14-slim-bookworm`, fixed by image digest. It mounts
nothing from the host. Follow the same commands below inside that shell. Direct
installation uses an empty working directory on a machine with Python 3.12;
run the first block as root (or prefix its `apt-get` commands with `sudo`).

<!-- FIRST_USE_COMMANDS_BEGIN -->
```bash
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git tar coreutils
rm -rf /var/lib/apt/lists/*
```

Run the remaining commands as one shell session:

```bash
set -eu
mkdir tubular-first-use
cd tubular-first-use
python3.12 -m venv .venv
git clone --depth 1 --branch v2.0.3 --recurse-submodules --shallow-submodules \
  https://github.com/icirauqui/LumenMorph.git
test "$(git -C LumenMorph rev-parse HEAD)" = 7d704d7471406affeb7e49c22dd2c057a62e2645
.venv/bin/python -m pip install --no-cache-dir -r LumenMorph/backend/requirements-collection.txt
mkdir downloads inputs results
DATA=https://huggingface.co/datasets/icirauqui/synthetic-deformable-tubular-benchmark/resolve/v2.0.3
GUIDE=https://huggingface.co/datasets/icirauqui/synthetic-deformable-tubular-benchmark/resolve/012319bdba646a6752d11ab13462876f9d976df7/first_use
curl --fail --location "$DATA/example.tar.gz" -o downloads/example.tar.gz
curl --fail --location "$GUIDE/check_first_use.py" -o downloads/check_first_use.py
curl --fail --location "$GUIDE/FIRST_USE_SHA256SUMS" -o downloads/FIRST_USE_SHA256SUMS
(cd downloads && sha256sum --check FIRST_USE_SHA256SUMS)
tar -xzf downloads/example.tar.gz -C inputs
export PYTHONPATH="$PWD/LumenMorph"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python downloads/check_first_use.py snapshot --root inputs/example --results results
for family in f07 f08; do
  for texture in procedural tissue; do
    stream="${family}_fem_moving_${texture}"
    .venv/bin/python LumenMorph/examples/consume_collection.py \
      --root inputs/example --stream "$stream" --frame 0 --target 1 \
      --output "results/$stream"
  done
done
.venv/bin/python downloads/check_first_use.py verify \
  --root inputs/example --results results --software LumenMorph
```
<!-- FIRST_USE_COMMANDS_END -->

Results are in `tubular-first-use/results/`, separate from `inputs/example/`.
Each stream directory contains `zero_displacement.npz` (a saved control
prediction), `score.json` (eligible population, coverage and errors), and
`example.json` (RGB dimensions and the queried material point's target position).
`FIRST_USE_REPORT.json` combines checks of all four routes;
`INPUT_SHA256.json` records input-file identities before the runs and is checked
again afterwards. The verification includes RGB channel order, all 12 images,
128 material queries and the exact local-to-canonical frame map.

Local frame indices 0, 1, 2 correspond to canonical frames 120, 121, 122 for f07
and 150, 151, 152 for f08. Material positions and displacement scores use
benchmark-world coordinates and scene-length units, not millimetres. The control
is saved before evaluator-only material data are opened. Its coverage is 1.0
because it supplies finite zeros at every eligible pixel; its **positive error**
measures actual motion missed by that control. This checks the reader and scoring
interfaces, not the accuracy of an image-based reconstruction method. Texture
pairs share material states and therefore have the same control scores.

Representative output from the recorded clean Docker run:

```text
f07_fem_moving_procedural: RGB 480x640x3; coverage=1.000000; mean endpoint error=0.001762968 scene_length
f07_fem_moving_tissue: RGB 480x640x3; coverage=1.000000; mean endpoint error=0.001762968 scene_length
f08_fem_moving_procedural: RGB 480x640x3; coverage=1.000000; mean endpoint error=0.002690495 scene_length
f08_fem_moving_tissue: RGB 480x640x3; coverage=1.000000; mean endpoint error=0.002690495 scene_length
PASS: 4 streams, 12 RGB reads, 128 material queries; 39 input files unchanged.
```

The [consumer records](https://huggingface.co/datasets/icirauqui/synthetic-deformable-tubular-benchmark-reproducibility/tree/v2.0.3-docs1/first_use)
retain the image identity, commands, logs, versions and results. This clean-environment
qualification was performed by the project team; an independent human usability
test has not been performed.
To repeat, use a new working directory/container; the example refuses to overwrite
existing per-stream result folders. Mechanical records, references and stationary
acquisitions require the complete collection below.

The [collection guide](docs/COLLECTION.md) describes the complete resource and its annotation contracts.

[Collection design, counts and evidence guide](docs/COLLECTION_GUIDE.md) defines families, actions, geometric acquisitions and RGB streams, and locates the validation and reconstruction records.

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

## Requirements for the browser workspace

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
Other dependencies retain their own licenses. Generated benchmark data
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

Create the environment on the current machine; copied Linux virtualenv paths
cannot run on macOS. Tests for the optional appearance renderer require its
Mitsuba/DrJit dependencies and an available LLVM shared library. If LLVM is not
discoverable, set `DRJIT_LIBLLVM_PATH` to the installed library; set
`DRJIT_CACHE_DIR` to a writable cache directory. The author-review qualification
used Mitsuba 3.9.1, DrJit 1.5.0 and LLVM 20.1.8 on macOS/aarch64. Missing optional
dependencies must be reported as omitted coverage, not a complete test pass.

The current CPU rasterizer clips near-plane crossing triangles while retaining
original material identity. The original sealed bank and its producer archive
remain unchanged; consult their conventions when replaying historical bytes.


## Auxiliary reconstruction execution

`python -m backend.src.benchmark_reconstruction --help` documents the calibrated
CPU SfM driver. `--timeout 0` (the default) lets every command finish naturally;
positive values impose an explicitly requested seconds-per-command budget.
A timed-out command is an interrupted computation, not evidence that a sequence
cannot be reconstructed. The driver retains every model and requested window.

For a sufficiently sized RAM filesystem or other fast scratch storage, pass
`--database-directory /path/to/new/job_database`. The directory must not exist.
Images, calibration and scientific options are unchanged. Logs, model outputs
and reports use `--output`; a hash-verified ordinary-file database checkpoint is
written there after each closed command. Do not copy or relocate the working
database during a command. The working directory can be retired only after the
persistent copy has been verified and the job has closed. Interrupted attempts
must be retained separately from subsequent executions.
