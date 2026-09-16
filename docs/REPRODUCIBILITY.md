# Reproducibility and scientific scope

## Record every generation

For a sequence or dataset you intend to share, retain:

1. The exact JSON configuration.
2. The seed and the config hash from `dataset_manifest.json`.
3. The LumenMorph Git commit.
4. `backend/uv.lock` and the Python version.
5. The operating system and CPU/GPU details when image-byte identity matters.
6. For FEM-derived suites, the `ContinuumFEM` source identity and the recorded
   `CONTINUUM_FEM_SRC` location.

Run the validator after generation:

```bash
backend/.venv/bin/python -m backend.src.cli validate --dataset /path/to/dataset
```

The validator checks required files, metadata, camera motion/clearance, sampled
feature quality and video readability. It does not establish clinical realism
or validate an external reconstruction method.

## Determinism

Identical code, configuration, seed, dependencies and platform are the target
for deterministic replay. A successful run on another operating system or a
different dependency build can still change rendered image bytes. Compare
declared hashes and the required scientific quantities; do not infer exact
cross-platform equality without checking it.

The tubular benchmark has stronger frozen provenance, source snapshots and QA.
Follow [TUBULAR_BENCHMARK.md](TUBULAR_BENCHMARK.md) when using that bank rather
than regenerating it casually.

## Mechanical interpretation

The generator records exact prescribed kinematics and, where configured,
mechanical-model outputs. Those values are ground truth **for the synthetic
scene**. They do not by themselves validate nonlinear tissue behaviour,
patient-specific mechanics, or clinical transfer. Preserve this distinction in
papers, dataset cards and method comparisons.

## Evaluation discipline

Never provide hidden truth fields, material coordinates, mesh states, or camera
poses to an estimator unless that information is explicitly part of its stated
input. The benchmark evaluator can use truth only after inference to calculate
coverage and error. See [EVALUATION.md](EVALUATION.md).
