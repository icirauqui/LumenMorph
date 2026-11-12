# Changelog

## 2.0.2 — 2026-11-12

- Explicit unit-sum decoding of float32 material coordinates for geometric
  queries and flow; existing datasets retain their recorded arithmetic.
- Read materialized reference acquisitions directly, while retaining support
  for historical linked layouts.
- Record geometric weight conventions and decoder identities in evaluation
  receipts; regression tests cover affine invariance and visibility.
- Independent accelerated full-pixel first-hit census and qualified native ray backend.
- Producer replay covers reference and every motion/camera condition at three times,
  both textures and next-frame flow, with the declared material-coordinate policy.
- Restore source-only freeze support for specifications without scene families.

## 2.0.1 — 2026-11-11

- Direct the colon-layout family's sigmoid toward the medial side after the
  descending segment, preserving the three-dimensional bends and acquisition
  parameterization. Add a centerline regression check for the intended layout.
- The six established family specifications, procedural textures and reader
  contracts remain unchanged. External textures are configured only for f07/f08.

## 2.0.0 — 2026-11-11

- Configurable long 3D centerlines, smooth radius profiles, parallel-transport
  frames, camera sampling, spatial basis counts and raised-cosine schedules.
- Hash-identified external material textures with periodic linear-light sampling;
  paired texture images share each acquisition's geometry, poses and truth.
- Explicit RGB collection/individual readers preserving historical rgb() contracts.
- Appearance dense scoring and one-component global Sim(3) material-track scoring.
- Analytic C3D6 checks, statewise surface/camera checks, stratified independent-ray
  validation, FE transfer checks and representative producer replay.
- Versioned collection configurations, CC0 texture provenance, examples and
  numerical environment pins. Existing published inputs remain immutable.

## 1.0.2 — 2026-10-10

- Tag the corrected code already present on `main`: material-preserving
  near-plane clipping, dense-truth handling, semantic benchmark selection and
  benchmark-flow identity fixes committed on 9 October.
- Align the backend project, its locked local-package record, the browser
  package and citation metadata with software version 1.0.2. Dependency versions
  and the independently versioned local API contract remain unchanged.
- Add a pinned recursive-clone command and the verified dataset DOI to the
  README. Dataset version 1.0.0 and its historical producer/source hashes are
  distinct from this corrected software release; no dataset is regenerated.


## 0.1.0 — 2026-09-16

- Established LumenMorph as the public name for the synthetic deformable
  endoscopic sequence generator and its local control surfaces.
- Documented installation, architecture, reproducibility, contribution and
  security policies.
- Standardized public branding across the CLI, API and browser workspace.
