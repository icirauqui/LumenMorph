# Tubular benchmark organization

Use [the collection guide](COLLECTION.md) for the consolidated resource and
[the evaluation contracts](EVALUATION.md) for scoring. `COLLECTION.json` is the
public index: select by scientific family, motion, camera and appearance rather
than infer those properties from numeric package directory names.

Each of eight seeded geometric families supplies a moving undeformed reference,
static/FE/non-FE motion and moving/stationary action cameras. f01–f06 have
90-frame acquisitions, f07 has 240, and f08 has 300. f07 and f08 have procedural
and file-texture images of each acquisition with shared geometry and truth.
Five path-traced appearance conditions use the moving-camera FE acquisitions
of f01, f04, f05 and f06. Appearance does not create independent geometries.

| Records | Location in the downloaded collection | Frame convention |
|---|---|---|
| f01–f06 raster acquisitions | `benchmark/v1` through `benchmark/v18` | 90 per acquisition |
| f07 raster acquisitions | `acquisitions/v19`, `v20`, `v21` | 240 per acquisition |
| f08 raster acquisitions | `acquisitions/v22`, `v23`, `v24` | 300 per acquisition |
| Path-traced appearances | `appearance/families/<family>` | 90 shared material states, five appearances |

Within each group of three packages, conditions are static, FE and non-FE.
Shared references are counted once. These package numbers identify motion
conditions, not successive public releases. A generator configuration may use
another package numbering; inspect its explicit index/descriptor rather than
copy a directory number between layouts.

`public/` contains inference observations and calibration. `evaluation/`
contains publicly downloadable privileged truth. Evaluator-only describes the
information contract, not access restrictions. Known-pose or supplied-geometry
experiments must declare those inputs. Estimated reconstructions retained with
some packages are method outputs, not ground truth.

The collection is a controlled synthetic resource. Its families and numerical
outputs have been inspected for generation and validation. Historical split
names in immutable metadata do not establish an unseen test set for a new
study. Freeze family-level splits and algorithm choices before outcomes; keep
paired cameras, motion conditions and appearances in the same partition.

All motion is prescribed from spatial fields and stored coefficients. The FE
fields solve discrete static linear-elastic systems with assigned parameters;
non-FE fields have no equilibrium claim. The volume is an auxiliary mechanical
discretization linked parametrically to the observation surface. It does not
establish patient anatomy or validated tissue biomechanics.

Preserve input files and their manifests; write predictions/results separately.
The reproduction deposit distinguishes regenerating images, rerunning checks,
replaying retained scores and historical inference. Use the exact source and
runtime identities attached to each declared reproduction claim.
