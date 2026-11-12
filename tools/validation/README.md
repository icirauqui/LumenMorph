# Independent first-intersection census

These optional validation tools trace every pixel-center ray against the stored
world-space triangles using double-precision Moller–Trumbore intersections and a
conservative bounding-volume hierarchy. They do not call the producer's rasterizer
or its continuous visibility query. C++17 is required for the ray backend; normal
reader use does not require this compilation.

From the repository root, with the numerical dependencies installed:

```bash
c++ -O3 -fPIC -shared -std=c++17 -ffp-contract=off \
  tools/validation/native_rays.cpp -o tools/validation/native_rays.so
python tools/validation/qualify_rays.py "$PWD"
python tools/validation/census.py --root "$SDTUBE_DATA/benchmark" \
  --family f07 --condition fem --stream moving --output "$SDTUBE_OUT/f07_fem_moving"
```

Use a fresh output directory. Repeat for each condition (`static`, `fem`,
`nonfem`) and camera (`moving`, `stationary`); the shared reference is
`--condition static --stream reference`. Select `f08` for that family.
`SPECIFICATION.json` in the public benchmark root supplies camera calibration,
package bindings and the recorded geometric material-weight policy. The optional
`--material-weight-policy` override is for an explicitly identified numerical
comparison, not an unrecorded change to the dataset contract.

The report records each frame's population, first-hit depth, material point,
reprojection, foreground/background disagreement, face identities, edge ties,
near-edge population, maxima and failed pixel IDs. Background and identity
criteria are exact except for the explicitly recorded shared-edge tie rule.
A background category with population zero is unexercised, not an independently
validated background example. The analytic qualification includes a missing-hit
case. Depth is camera z, not Euclidean distance along a normalized ray.

No data are changed. The companion contains exact execution code, reports,
configuration identities and environment records for the published census.
