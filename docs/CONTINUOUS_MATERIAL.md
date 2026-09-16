# Continuous reference-coordinate material appearance

This opt-in forward-rendering path evaluates an assigned Fourier reflectance field **after** interpolating material reference coordinates at a ray hit. It removes the need to resolve each texture wavelength with geometry vertices. It does not supply pixel antialiasing, measured tissue optics or resolved vascular geometry.

```python
from backend.src.continuous_material import seeded_fourier_field
from backend.src.endoscopic_renderer import render_endoscopic

# Choose once, then save this complete spectrum and the reference mapping.
field = seeded_fourier_field(reference_vertices_m.mean(axis=0), seed=20260911)
field_specification = field.to_dict()

rgb, truth, metadata = render_endoscopic(
    deformed_vertices_m, triangles, None, None,
    camera, R_wc, C_world, settings,
    continuous_material=field,
    material_reference_vertices_meter=reference_vertices_m,
)
```

`reference_vertices_m` must retain the same material vertex IDs as each geometry state. Do not replace it with deformed world positions. Freeze the field origin as well: recomputing an unweighted vertex mean after subdivision can change the origin and therefore the field. Save the explicit spectrum rather than relying only on a seed or a future library's random-number behavior. `FourierMaterialField.from_dict()` restores it; `sha256()` supplies its canonical identity.

The field is

`albedo(X) = base_rgb × [1 + Σ amplitude[j] cos(k[j] · (X − origin) + phase[j])]`.

`k` is in radians per reference meter. The amplitude bound ensures finite linear reflectance within [0,1]. Random directions and phases follow the original vertex-field generator's order, so the seeded field agrees with its multiscale-only values at the original vertices. Network attenuation is not part of this continuous field: the graph network remains a separate vertex-discrete approximation.

Reference coordinates are centered at the fixed field origin before conversion to float32 mesh attributes. Mitsuba interpolates those coordinates barycentrically and evaluates the spectrum in float32. Independent NumPy truth evaluates the same spectrum in float64 on reconstructed barycentric coordinates. Returned truth adds `material_reference_position_meter` and `diffuse_albedo_linear_rgb`; metadata includes the complete field, spectrum identity, reference-coordinate hashes and precision convention. Small renderer/truth differences are numerical and must be measured for the intended wavelength/coordinate range. These routines do not promise arbitrary-frequency precision.

Bitmap albedo, vertex albedo and continuous material are mutually exclusive. Existing bitmap and vertex behavior remains available. The continuous field is attached to an ambient reference XYZ frame; it does not turn that frame into an intrinsic/geodesic tissue coordinate system. Nearby reference-space surfaces can share correlated field values.

The roughplastic adapter returns the base RGB arithmetic mean as a phase-ensemble **sampling proxy**, not an exact spatial area mean. Mitsuba uses that number to choose diffuse/specular proposal weights, while local field evaluation remains spatially varying. It is independent of tessellation and geometry state and is exact for the zero-mode constant control. There is no inverse-rendering or differentiability claim for the custom texture adapter.

Contract tests cover serialized replay, bounded inputs, agreement with the old vertex samples, constant bitmap/vertex controls, analytic Lambertian radiometry, coarse/subdivided planes at fixed rays, deformation attachment and roughplastic proposal semantics. Coarse/subdivided invariance applies when geometry and the piecewise reference mapping describe the same surface; it does not remove geometric approximation error for curved anatomy. Pixel-center sampling still aliases unresolved image-space detail. No anatomical renders or population qualification are implied by these tests.
