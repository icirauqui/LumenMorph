# First-use guide

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

The [consumer records](https://huggingface.co/datasets/icirauqui/synthetic-deformable-tubular-benchmark-reproducibility/tree/v2.0.3-consumer1/first_use)
retain the image identity, commands, logs, versions and results. This clean-environment
qualification was performed by the project team; an independent human usability
test has not been performed.
To repeat, use a new working directory/container; the example refuses to overwrite
existing per-stream result folders. Mechanical records, references and stationary
acquisitions require the complete collection below.
