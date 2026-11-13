# Synthetic Deformable Tubular Benchmark — collection guide

## Collection design and terminology

The eight families form one collection. A **family** fixes a seeded tubular
geometry, material coordinates, camera paths and spatial displacement bases.
An **action** combines a motion condition (static, FE or non-FE) with a moving
or stationary camera. Non-FE means displacement prescribed directly, without
solving an FE equilibrium problem. A separate moving-camera reference scans
the undeformed surface.

A **geometric acquisition** is a sequence of states, cameras and geometric
annotations. An **RGB stream** renders that acquisition with a specified
texture and rendering condition. Texture variants share geometric annotations;
they are not independent geometries.

| Component | Families and scope | Geometric acquisitions | RGB streams | RGB images |
|---|---|---:|---:|---:|
| Raster, procedural | f01–f06; reference and six actions per family; 90 frames | 42 | 42 | 3,780 |
| Raster, paired procedural/organic | f07/f08; reference and six actions per family; 240/300 frames | 14 | 28 | 7,560 |
| Raster total | All eight families | 56 | 70 | 11,340 |
| Path-traced | Moving FE actions of f01/f04/f05/f06; five lighting/reflectance/exposure conditions, 90 frames | Four shared trajectories, converted as documented | 20 | 1,800 |
| Collection total | Eight engineered geometries | 56 raster acquisitions; path-traced states identified separately | 90 | 13,140 |

Raster rendering uses triangle projection and a depth buffer. Path tracing
simulates light transport; its five conditions share states within that component.
Its metre conversion, rounded vertices and projected rotations mean that a
raster/path-traced comparison is not exclusively a photometric intervention.
Only f07/f08 have paired procedural/organic textures in the raster pipeline.
The collection has 7,560 raster dense maps and 7,504 forward-flow maps, stored
once per geometric acquisition rather than once per texture.

All raster images are 640×480. f01–f06 use 3 motion bases and 5,376 rendering
vertices; f07 uses 6 bases and 30,720 vertices; f08 uses 8 and 36,864. Stored
basis and frame indices are zero-based. With T frames, i/(T−1) is normalized
action time. FE bases solve static linear systems; temporal coefficients
prescribe their combination, not dynamic mass/damping/inertia integration.
The observation surface is transferred parametrically from the outer FE-node
envelope; the auxiliary mechanical volume extends inward. This is not a
validated anatomical wall model. Mechanical strain/stress are FE quantities,
not per-pixel labels.

Raster `unresolved` is a source-valid point whose target projection is finite,
positive-depth and inside the image, but neither visible nor occluded. Given
projected camera-z depth z and nearest target surface depth d, visibility uses
`abs(z-d) <= 1e-5 + 1e-5*abs(z)`; occlusion uses `z > d + tolerance`.
An unresolved query has no finite target hit or lies ahead of it beyond that
tolerance. It is not automatically an occlusion. f07/f08 contain zero such
labels; zero-population categories are unexercised. The path-traced component
does not supply this raster visibility classification.

## Locating evidence and reproducing checks

- Dataset `COLLECTION.json`: streams, geometric acquisitions, modalities and paths.
- Dataset `RECONSTRUCTION_INDEX.json`: 40 global and 486 local SfM products,
  registered-image populations and model/window mappings.
- Companion `validation/FAMILY_COVERAGE.json`: per-family coverage matrix.
- Companion `validation/CHECK_INDEX.json`: reports, checked inputs, code identities,
  environments and reproducible commands. Census, sample and logical-consistency
  checks have distinct scopes; omitted and zero-population cases are explicit.
- Companion `producers/`, `software/`, `dataset_records/` and `commands/`:
  exact generation sources, numerical software, configurations and entry points.
- Companion `execution/replay/`, `execution/consumer/` and `first_use/`:
  representative producer replay, per-family modality controls and clean public
  consumption. These checks are not exhaustive regeneration of every image.
- Companion `execution/clean_consumer_20261014/` and
  `execution/CLEAN_CONSUMER_IMPORT.json`: installation evidence with exact provenance.
- Companion `studies/`: optional associated application-study archives. Missing
  study intermediates limit those studies' replay, not the collection's recipes.

The auxiliary reconstructions use the colonoscopy fork of **COLMAP 3.7**, archived
CPU executable reporting commit `0f7e57c`. Each job's `provenance/manifest.json`
and `provenance/executable_help.txt` identify the binary and runtime; captured
source is a separate provenance fact. The protocol fixes calibration, seed 0,
two threads, 12,000 SIFT features, maximum image dimension 800 and 15-frame local
windows at stride 10 plus the final boundary window. Registration coverage does
not establish accurate reconstruction. See Schönberger and Frahm, *Structure-
from-Motion Revisited*, CVPR 2016, https://doi.org/10.1109/CVPR.2016.445.

A **reconstruction component** is a coordinate-connected set of estimated cameras
and surface entities sharing one global gauge. Its `component_id` identifies
that set in a monocular evaluation protocol; it does not mean raster/path-traced
rendering component. Independently aligned sets are not a joint reconstruction.
The auxiliary **prefix-anchor diagnostic** fits one similarity using the earliest
`max(3, ceil(0.2*N))` registered expected camera centers in frame order and applies
it to the entire model trajectory (N is the registered expected-camera count).
It reports trajectory/endpoint errors and endpoint error divided by true path
length through registered cameras. It probes sensitivity to early anchoring,
separately from the primary all-anchor fit; degenerate fits are unavailable.
All truth-aligned results are evaluation diagnostics, not absolute-scale recovery.

## Choosing the consumption example

The full f01 FE example opens `benchmark/v2` and queries frames 52–53. It requires
the complete f01 FE package and does not run on the compact example. The compact
f07/f08 archive uses `examples/consume_collection.py` with local frames 0–2,
corresponding to canonical 120–122 and 150–152 respectively. Follow the first-use
README commands for that route. `Collection.image_rgb()` always returns RGB;
`PublicDataset.rgb()` retains BGR and `AppearancePublic.rgb()` retains RGB.
