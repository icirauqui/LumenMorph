# Synthetic Deformable Tubular Benchmark

Start with the [first-use guide](FIRST_USE.md) to verify the compact public example, read both families/textures, query material and score a saved control. It includes direct installation and a pinned Docker route; no full dataset download or Hub credentials are needed.

The collection index `COLLECTION.json` identifies every RGB acquisition by family,
motion, camera and appearance. A family is a distinct seeded geometry. Texture
and illumination variants reuse geometric states; they are not independent
anatomical samples. All images are 640×480. Lengths in the raster acquisitions
are abstract scene units; the path-traced appearance records use their documented
metre conversion.

## Install the numerical tools

From a recursive LumenMorph checkout, create a separate environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-collection.txt
.venv/bin/python -m pytest tests/test_collection.py tests/test_collection_evaluation.py -q
```

The mechanics source is the pinned `continuum-fem` Git submodule. No browser,
Node.js, GPU or access to evaluation data is needed to read RGB images.

[Collection design, counts and evidence guide](COLLECTION_GUIDE.md) defines families, actions, geometric acquisitions and RGB streams, and locates the validation and reconstruction records.

## Read images

Set `SDTUBE_DATA` to the extracted collection root containing `COLLECTION.json`.
The same interface opens the independently downloadable small example.

```python
import os
from backend.src.collection_reader import Collection

collection = Collection(os.environ['SDTUBE_DATA'])
streams = collection.select(family='f07', motion='fem', camera='moving')
image = collection.image_rgb('f07_fem_moving_tissue', 0)
```

`image_rgb()` returns uint8 RGB in every reader. `PublicDataset.rgb()` retains
its BGR contract; `AppearancePublic.rgb()` retains RGB. The explicit RGB method
avoids inferring channel order from the historical name. Each texture variant
uses the exact same camera, geometry, material weights and annotations within
its acquisition. The image-only reader never reads evaluation files.

## Query a material point and score a prediction

```bash
PYTHONPATH=. .venv/bin/python examples/consume_collection.py \
  --root "$SDTUBE_DATA" --output ./example_results
```

Run this example from the checkout with `PYTHONPATH=.` if invoking the script
by filename. It writes the zero-displacement prediction before opening ground
truth, then queries one persistent material point and evaluates the prediction.
It demonstrates the interface, not an image-based reconstruction method. Write
results outside the read-only dataset and use a fresh output directory.

Dense raster scoring uses `backend.src.benchmark_evaluate`: depth (camera z),
oriented normals, material displacement and occlusion-aware next-frame flow.
The scorer preserves the requested truth population when predictions are absent.
See [evaluation contracts](EVALUATION.md).

## File textures

The family configuration's `file_textures` maps names to a relative image path,
SHA-256, `orientation` (counter-clockwise 0/90/180/270 degrees), positive UV
`repeat`, `offset`, `interpolation`, `color_space='srgb'`, `seam='periodic'` and
optional recorded randomization seed. Images are sampled on persistent material
UV, not on camera coordinates or deformed positions. Bilinear interpolation
operates in linear-light RGB before encoding to sRGB; nearest mode is also
supported. Wrapping includes interpolation across both periodic image edges.
Material triangles spanning the angular seam are unwrapped before interpolation.
This guarantees periodic sampling, not visual continuity of an arbitrary image.

The supplied TextureCan Others 0001 color map is CC0 and procedurally made. It
provides stylized organic color, not measured tissue reflectance. Only its color
map is used; its normal, height, ambient-occlusion and roughness files are not
applied. Original files, hashes and license/provenance accompany the resource.
Both procedural and file-texture images receive identical deterministic noise
and gain draws. Geometry/truth is stored once for their shared acquisition.

## Generate an acquisition collection

```bash
.venv/bin/python -m backend.src.benchmark freeze \
  --spec configs/tubular_collection.json --root ./generated_collection
.venv/bin/python -m backend.src.benchmark prepare \
  --root ./generated_collection --family f07
.venv/bin/python -m backend.src.benchmark package \
  --root ./generated_collection --family f07 --condition fem
.venv/bin/python -m backend.src.benchmark render \
  --root ./generated_collection --family f07 --condition fem --stream moving
```

Use a fresh root. Freezing captures the specification, source hashes, runtime
and external image assets. The reproduction deposit includes the exact producer
archive, complete commands and reports for all released acquisitions. Modern
source documentation and frozen producer identities have separate roles: use
the recorded producer archive and runtime to test regeneration equality.

`collection_geometry` selects smooth radius profiles and parallel-transport
frames. `reference_frames`, `action_frames`, the number/localization of patches
and the `temporal_coefficients` list are configurable. The list must have one
entry per spatial basis. Raised-cosine pulses are zero outside their declared
normalized start/end times. FE bases solve static linear systems once; their
prescribed temporal combination is not mass/damping/inertia integration.
The configurations without these options keep their original 90-frame schedule.

The observation surface is parametrically linked to the exterior FE-node
sampling envelope. The auxiliary FE volume extends inward from that envelope.
It is not a physical intestinal wall with a validated luminal inner boundary.
Mechanical stress/strain belongs to FE quadrature, not directly to image pixels.

Generation does not certify a new configuration. Run the recorded mechanical,
surface/camera and stratified label checks before using it as evaluation truth.
