"""Explicit decoding of stored material coordinates; no silent legacy change."""
import numpy as np


def decode_weights(values, policy='stored'):
    """Promote weights, optionally restoring their affine partition of unity.

    ``unit_sum`` removes storage-roundoff in the sum before geometric queries.
    It is translation invariant and keeps the point on its material triangle.
    ``stored`` preserves previously published arithmetic. Texture sampling has
    its own recorded convention and is not changed by this geometric policy.
    """
    weights = np.asarray(values, dtype=np.float64)
    if policy not in ('stored', 'unit_sum'):
        raise ValueError('Unknown material weight policy')
    if policy == 'stored':
        return weights
    total = weights.sum(axis=-1, keepdims=True)
    if (weights.shape[-1] != 3 or not np.isfinite(weights).all()
            or np.any(weights < -1e-6)
            or not np.allclose(total, 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid physical material weights')
    return weights / total
