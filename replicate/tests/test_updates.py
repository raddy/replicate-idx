import numpy as np
import pytest
from ..optimizers.updates import ETEUpdate

def test_ete_update_compute_matrices():
    """Test matrix computation for ETE updates."""
    updater = ETEUpdate()
    
    # Setup test data
    m, n = 10, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    
    matrices = updater.compute_matrices(X, r)
    
    assert 'B' in matrices, "Should contain B matrix"
    assert 'b' in matrices, "Should contain b vector"
    assert 'Lmax_A' in matrices, "Should contain largest eigenvalue"
    
    assert matrices['B'].shape == (n, n), "B should be n x n"
    assert matrices['b'].shape == (n,), "b should be n-dimensional"
    assert np.isscalar(matrices['Lmax_A']), "Lmax_A should be scalar"

def test_ete_update_step():
    """Test single update step."""
    updater = ETEUpdate()
    
    # Setup test data
    m, n = 10, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    w = np.array([0.5, 0.3, 0.2])
    
    # Get matrices
    matrices = updater.compute_matrices(X, r)
    
    # Perform update
    w_new = updater.update(
        w=w,
        lambda_=1e-7,
        p=1e-3,
        c1=1.0,
        u=0.5,
        **matrices
    )
    
    assert np.isclose(np.sum(w_new), 1.0), "Updated weights should sum to 1"
    assert np.all(w_new >= 0), "Updated weights should be non-negative"
    assert np.all(w_new <= 0.5), "Updated weights should respect upper bound"

def test_ete_update_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    updater = ETEUpdate()
    
    # Test invalid matrix computation inputs
    with pytest.raises(AssertionError):
        # Wrong dimension for X
        updater.compute_matrices(np.random.randn(10), np.random.randn(10))
    
    with pytest.raises(AssertionError):
        # Mismatched dimensions
        updater.compute_matrices(np.random.randn(10, 3), np.random.randn(5))
    
    # Test invalid update inputs
    m, n = 10, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    matrices = updater.compute_matrices(X, r)
    
    with pytest.raises(AssertionError):
        # Wrong dimension for w
        updater.update(
            w=np.random.randn(n, 1),  # Should be 1D
            lambda_=1e-7,
            p=1e-3,
            c1=1.0,
            u=0.5,
            **matrices
        )

def test_ete_update_convergence():
    """Test that updates improve the objective."""
    updater = ETEUpdate()
    
    # Setup test data
    m, n = 10, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    w = np.ones(n) / n  # Start with uniform weights
    
    # Get matrices
    matrices = updater.compute_matrices(X, r)
    
    # Perform multiple updates
    tracking_errors = []
    for _ in range(5):
        w = updater.update(
            w=w,
            lambda_=1e-7,
            p=1e-3,
            c1=1.0,
            u=0.5,
            **matrices
        )
        tracking_errors.append(np.linalg.norm(X @ w - r))
    
    # Check that tracking error generally decreases
    assert tracking_errors[-1] <= tracking_errors[0], \
           "Tracking error should decrease with updates"
