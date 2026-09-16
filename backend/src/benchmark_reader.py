"""Public-input and explicitly evaluator-only access to tubular benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .dense_truth import flow_to_frame
from .types import CameraIntrinsics


class PublicDataset:
    """Read only the declared public sensor/acquisition inputs.

    Construct with a version package (e.g. ``.../v4``), not the suite root.
    This class never opens an evaluation file or a ground-truth pose.
    """

    def __init__(self, package: str | Path):
        self.package = Path(package)
        self.descriptor = json.loads((self.package / "DATASET.json").read_text())
        self.intrinsics = CameraIntrinsics(**json.loads((self.package / "public/intrinsics.json").read_text()))

    def public_paths(self, stream: str = "stationary") -> dict[str, Path]:
        self._check_stream(stream)
        return {
            "rgb": self.package / "public" / stream / "rgb",
            "frames": self.package / "public" / stream / "frames.json",
            "intrinsics": self.package / "public/intrinsics.json",
            "acquisition": self.package / "public/acquisition.json",
            "reference_rgb": self.package / "public/reference/rgb",
            "reference_frames": self.package / "public/reference/frames.json",
        }

    @staticmethod
    def _check_stream(stream):
        if stream not in {"reference", "moving", "stationary"}:
            raise ValueError("stream must be reference, moving, or stationary")

    def frames(self, stream: str = "stationary") -> list[dict]:
        return json.loads(self.public_paths(stream)["frames"].read_text())["frames"]

    def _frame(self, stream, frame):
        frames = self.frames(stream)
        if not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(frames):
            raise IndexError(f"Frame {frame!r} is outside {stream}")
        return frames[int(frame)]

    def rgb(self, frame: int, stream: str = "stationary") -> np.ndarray:
        """Load uint8 BGR, matching OpenCV and generator channel order."""
        record = self._frame(stream, frame)
        path = self.public_paths(stream)["rgb"] / record["image"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return image


class EvaluationDataset(PublicDataset):
    """Explicitly privileged reader. Never pass its truth outputs to inference.

    Material points use immutable triangle IDs plus physical barycentric weights;
    the same identities can be queried in every stream and time state.
    """

    def truth_path(self, stream: str = "stationary") -> Path:
        self._check_stream(stream)
        if stream == "reference":
            return (self.package / "evaluation/family").resolve().parent / "reference_truth"
        return self.package / "evaluation" / stream

    def geometry(self) -> dict[str, np.ndarray]:
        with np.load(self.package / "evaluation/family/geometry.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def state(self, frame: int, stream: str = "stationary") -> dict[str, np.ndarray]:
        self._frame(stream, frame)
        with np.load(self.truth_path(stream) / "states.npz", allow_pickle=False) as data:
            return {key: data[key][frame] for key in data.files}

    def dense(self, frame: int, stream: str = "stationary") -> dict[str, np.ndarray]:
        self._frame(stream, frame)
        with np.load(self.truth_path(stream) / "dense" / f"{frame:06d}.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def material_points(
        self, triangle_ids: np.ndarray, barycentric: np.ndarray,
        frame: int | None = None, stream: str = "stationary",
    ) -> np.ndarray:
        """Return world XYZ. ``frame=None`` selects the static reference mesh.

        Indices may have any shape; barycentrics must have that shape plus a
        final length-three dimension. These are physical surface weights,
        never affine image-space weights. Invalid/background IDs are rejected.
        """
        ids = np.asarray(triangle_ids)
        bary = np.asarray(barycentric, dtype=np.float64)
        geometry = self.geometry()
        triangles = geometry["triangles"]
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Triangle IDs must be integers")
        if bary.shape != ids.shape + (3,):
            raise ValueError("Barycentric shape must equal triangle ID shape plus (3,)")
        if np.any(ids < 0) or np.any(ids >= len(triangles)):
            raise ValueError("Triangle ID outside persistent topology")
        if not np.all(np.isfinite(bary)) or np.any(bary < -1e-6) or not np.allclose(bary.sum(axis=-1), 1., atol=1e-6, rtol=0):
            raise ValueError("Expected finite nonnegative barycentrics summing to one")
        vertices = geometry["vertices"] if frame is None else self.state(frame, stream)["vertices"]
        return np.sum(bary[..., :, None] * vertices[triangles[ids]], axis=-2)

    def material_displacement(
        self, triangle_ids: np.ndarray, barycentric: np.ndarray,
        frame: int, stream: str = "stationary", from_frame: int | None = None,
    ) -> np.ndarray:
        """Exact material displacement from reference or another same-stream time."""
        target = self.material_points(triangle_ids, barycentric, frame, stream)
        source = self.material_points(triangle_ids, barycentric, from_frame, stream)
        return target - source

    def compute_flow(self, source_frame: int, target_frame: int, stream: str = "stationary") -> dict[str, np.ndarray]:
        """Derive forward, backward, or long-range flow and visibility on demand.

        This recomputes from material identity and exact target geometry, using
        the same qualified continuous visibility query as stored forward flow.
        No files are written and no estimator correspondence is substituted.
        """
        dense = self.dense(source_frame, stream)
        target = self.state(target_frame, stream)
        conventions = json.loads((self.truth_path(stream)/'CONVENTIONS.json').read_text())
        return flow_to_frame(dense, target["vertices"], self.geometry()["triangles"],
                             self.intrinsics, target["R_wc"], target["C_world"],
                             high_precision=conventions.get('high_precision', False))

    def mechanics(self, frame: int, stream: str = "stationary") -> dict[str, np.ndarray | float | str]:
        """Compose evaluator-only FE fields at an acquisition frame.

        FE nodes and quadrature fields describe the coarse generating model,
        not the interpolated render mesh or validated tissue mechanics. Energy
        includes all cross terms: 0.5*u.T*K*u. Basis energies cannot be summed.
        Non-FEM actions have no applied constitutive/load truth and are rejected.
        Static actions and all rigid references return zero fields.
        """
        self._frame(stream, frame)
        condition = self.descriptor['condition']
        if stream != 'reference' and condition == 'nonfem':
            raise ValueError('Non-FEM action has prescribed kinematics but no applied FE mechanics')
        with np.load(self.package/'evaluation/family/mechanics.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        coefficients = np.zeros(arrays['basis_displacement'].shape[0], dtype=np.float64)
        if stream != 'reference' and condition == 'fem':
            with np.load(self.package/'evaluation/family/fields.npz', allow_pickle=False) as data:
                coefficients = data['time_coefficients'][frame].astype(np.float64)
        result = {key: arrays[key] for key in ('nodes', 'elements_c3d6', 'fixed', 'E', 'nu')}
        for name in ('displacement', 'loads', 'reactions', 'gauss_strain', 'gauss_stress'):
            result[name] = np.tensordot(coefficients, arrays['basis_'+name], axes=(0, 0))
        from scipy.sparse import csr_matrix
        stiffness = csr_matrix((arrays['K_data'], arrays['K_indices'], arrays['K_indptr']), shape=tuple(arrays['K_shape']))
        displacement = result['displacement'].reshape(-1)
        result['strain_energy'] = float(.5 * displacement @ (stiffness @ displacement))
        result['coefficients'] = coefficients
        result['units'] = 'abstract scene length; numerically consistent force/stress/energy, no SI or tissue calibration'
        return result
