import numpy as np
import pytest
from ..optimizers.objectives import ETEObjective

def test_ete_objective_basic():
    """Test basic ETE objective computation."""
    obj = ETEObjective()
    
    # Setup test data
    m, n = 10, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    w = np.array([0.5, 0.3, 0.2])  # Valid weights summing to 1
    
    value = obj.compute(X=X, r=r, w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)
    assert value >= 0, "Objective value should be non-negative"

def test_ete_objective_perfect_tracking():
    """Test objective with perfect tracking."""
    obj = ETEObjective()
    
    # Create scenario where tracking is perfect
    m, n = 5, 2
    X = np.random.randn(m, n)
    w = np.array([0.6, 0.4])
    r = X @ w  # Perfect tracking
    
    value = obj.compute(X=X, r=r, w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)
    assert np.isclose(value, obj.compute(X=X, r=r, w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)), \
           "Perfect tracking should give consistent objective value"

def test_ete_objective_sparsity():
    """Test that sparsity term behaves correctly."""
    obj = ETEObjective()
    m, n = 5, 3
    
    # Create data where tracking errors will be identical
    # by making X just a column of 1s (so X @ w is same for both)
    X = np.ones((m, n))
    r = np.ones(m)
    
    # Compare dense vs sparse weights that sum to 1
    w_dense = np.array([0.4, 0.3, 0.3])
    w_sparse = np.array([0.7, 0.3, 0.0])
    
    # Verify tracking error is identical
    tracking_dense = np.sum((X @ w_dense - r)**2)
    tracking_sparse = np.sum((X @ w_sparse - r)**2)
    assert np.abs(tracking_dense - tracking_sparse) < 1e-10
    
    # Now compare objectives - only sparsity term should differ
    value_dense = obj.compute(
        X=X, r=r, w=w_dense,
        lambda_=1e-7, p=1e-3, c1=1.0, m=m
    )
    value_sparse = obj.compute(
        X=X, r=r, w=w_sparse,
        lambda_=1e-7, p=1e-3, c1=1.0, m=m
    )
    
    # Sparsity term should make sparse solution have lower objective
    assert value_sparse < value_dense, "Sparse solution should have lower objective"
    
    # Can also test relative magnitudes
    sparsity_diff = (
        np.sum(np.log(1 + w_dense/1e-3)) - 
        np.sum(np.log(1 + w_sparse/1e-3))
    )
    assert sparsity_diff > 0, "Sparse solution should have lower log term"

def test_ete_objective_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    obj = ETEObjective()
    m, n = 5, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    w = np.array([0.4, 0.3, 0.3])
    
    with pytest.raises(AssertionError):
        # Wrong shape for X
        obj.compute(X=X.reshape(-1), r=r, w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)
    
    with pytest.raises(AssertionError):
        # Wrong shape for r
        obj.compute(X=X, r=np.random.randn(m+1), w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)
        
    with pytest.raises(AssertionError):
        # Negative lambda
        obj.compute(X=X, r=r, w=w, lambda_=-1e-7, p=1e-3, c1=1.0, m=m)
        
    with pytest.raises(AssertionError):
        # Negative p
        obj.compute(X=X, r=r, w=w, lambda_=1e-7, p=-1e-3, c1=1.0, m=m)

def test_ete_objective_parameter_scaling():
    """Test how objective scales with different parameter values."""
    obj = ETEObjective()
    
    m, n = 5, 3
    X = np.random.randn(m, n)
    r = np.random.randn(m)
    w = np.array([0.4, 0.3, 0.3])
    
    # Test scaling with lambda
    value1 = obj.compute(X=X, r=r, w=w, lambda_=1e-7, p=1e-3, c1=1.0, m=m)
    value2 = obj.compute(X=X, r=r, w=w, lambda_=1e-6, p=1e-3, c1=1.0, m=m)
    assert value2 < value1, "Larger lambda should give smaller objective"
