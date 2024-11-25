import numpy as np
import pytest
from ..optimizers.kkt import project_weights_kkt

def test_project_weights_basic():
    """Test basic projection with no upper bound constraints."""
    c = np.array([0.1, 0.2, 0.3, 0.4])
    u = 0.5
    w = project_weights_kkt(c, u)
    
    assert np.isclose(np.sum(w), 1.0), "Weights should sum to 1"
    assert np.all(w >= 0), "All weights should be non-negative"
    assert np.all(w <= u), "All weights should be <= u"
    
def test_project_weights_upper_bound():
    """Test projection with active upper bound constraints."""
    c = np.array([-0.5, -0.4, 0.1, 0.2])
    u = 0.4
    w = project_weights_kkt(c, u)
    
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= 0)
    assert np.all(w <= u)
    assert np.any(np.isclose(w, u)), "At least one weight should hit upper bound"

def test_project_weights_edge_cases():
    """Test edge cases for weight projection."""
    # Test with uniform input
    c = np.ones(5)
    u = 0.5
    w = project_weights_kkt(c, u)
    assert np.allclose(w, 0.2), "Uniform input should give uniform output"

    # Test with very small u where sum-to-one is impossible
    c = np.array([0.1, 0.2, 0.3])
    u = 0.1
    w = project_weights_kkt(c, u)
    assert np.all(w <= u + 1e-10), "All weights should be at or below u"
    assert np.isclose(np.sum(w), 0.3), "Weights can't sum to 1, because they obey the cap of 0.1 (u)"
    
    # Test with u where sum-to-one is barely possible
    c = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    u = 0.25
    w = project_weights_kkt(c, u)
    assert np.all(w <= u + 1e-10), "All weights should be at or below u"
    assert np.isclose(np.sum(w), 1.0), "Weights should sum to 1"

def test_project_weights_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(AssertionError):
        # Test 2D input
        project_weights_kkt(np.ones((2, 2)), 0.5)
    
    with pytest.raises(AssertionError):
        # Test negative u
        project_weights_kkt(np.ones(3), -0.1)
        
    with pytest.raises(AssertionError):
        # Test u > 1
        project_weights_kkt(np.ones(3), 1.5)

def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    # Test with very large numbers
    c = np.array([1e5, 1e6, 1e7])
    u = 0.5
    w = project_weights_kkt(c, u)
    assert np.isclose(np.sum(w), 1.0)
    
    # Test with very small numbers
    c = np.array([1e-5, 1e-6, 1e-7])
    w = project_weights_kkt(c, u)
    assert np.isclose(np.sum(w), 1.0)
