import numpy as np
POWERS = (1,2,4)
def reliable_weights(risk, power):
    risk = np.asarray(risk, dtype=np.float64)
    assert risk.ndim == 1 and len(risk) > 0
    assert np.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all()
    assert power in POWERS
    retained = np.clip(1 - risk, .05, 1) ** power
    weights = retained / retained.mean()
    # Exact constant-weight identity permits honest reuse of the reference fit.
    if np.all(retained == retained[0]):
        weights = np.ones_like(weights)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert abs(weights.mean() - 1) < 1e-12
    return weights


def class_unit_mean(weights, observed):
    """Preserve each OBSERVED class's total coefficient, not its true-class mass."""
    weights = np.asarray(weights, dtype=np.float64)
    observed = np.asarray(observed)
    assert weights.ndim == 1 and len(weights) > 0
    assert observed.shape == weights.shape
    assert np.issubdtype(observed.dtype, np.integer) and (observed >= 0).all()
    assert np.isfinite(weights).all() and (weights > 0).all()
    normalized = weights.copy()
    for label in np.unique(observed):
        take = observed == label
        original = weights[take]
        normalized[take] = original / original.mean()
        # Keep exact identity for constant/singleton classes to allow safe reuse.
        if np.all(original == original[0]):
            normalized[take] = 1.
        np.testing.assert_allclose(normalized[take].sum(), int(take.sum()), rtol=1e-12, atol=1e-12)
    assert np.isfinite(normalized).all() and (normalized > 0).all()
    assert abs(normalized.mean() - 1) < 1e-12
    return normalized

