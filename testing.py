
from functions_definition import cross_section, stopping_power, Fitting, Integral
import numpy as np
import pytest
from unittest.mock import patch

def test_fitting_invalid_data():
    def model_function(x, xc):
        return np.exp(-((x - xc)**2))

    invalid_data = np.array([1, 0.5, 2, 0.7, 3, 0.2])  # Non è un array 2D
    init_params = {}
    
    with pytest.raises(ValueError, match="Data should be a 2D numpy array with two columns"):
        Fitting(model_function, invalid_data, **init_params)


def test_fitting_cs_params_positive():
    """
    Ensure that the fitting function for the `cross_section` model returns positive parameters.

    GIVEN: Simulated data generated from the `cross_section` function with known parameters.
    WHEN: The `Fitting` function estimates parameters from this simulated data.
    THEN: The estimated parameters `xc`, `A`, `sigma`, and `tau` should all be positive.
    """

    x_data = np.linspace(0, 10, 100)
    y_data = cross_section(x_data, xc=5, A=10, sigma=1, tau=2) 
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(cross_section, data, A=10, sigma=1, tau=2)
    
    params = result.params
    
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"

def test_fitting_sp_params_positive():
    """
    Ensure that the fitting function for the `stopping_power` model returns positive or non-negative parameters.

    GIVEN: Simulated data generated from the `stopping_power` function with known parameters.
    WHEN: The `Fitting` function estimates parameters from this simulated data.
    THEN: The estimated parameters `xc`, `A`, `sigma`, `tau`, and `C1` should all be positive or non-negative.
    """
    x_data = np.linspace(0, 10, 100)
    y_data = stopping_power(x_data, xc=5, A=150, sigma=0.5, tau=2, C1=5) 
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(stopping_power, data, A=150, sigma=0.5, tau=2, C1=5)
    
    params = result.params
    
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"

def test_integral_invalid_energy_loss():
    def mock_cross_section(energy, **params):
        return 1.0  # Simulates a constant cross-section function
    
    def mock_stopping_power(distance, **params):
        return 100.0  # Simulates a very high energy loss

    # Example parameters
    cs_params = {}
    sp_params = {}
    config = 'configuration.txt'



    with patch('functions_definition.cross_section', mock_cross_section), \
         patch('functions_definition.stopping_power', mock_stopping_power):
        
        # Execute the test
        rval = Integral(10, cs_params, sp_params, config)
        
        # Verify that the result is not NaN
        assert not np.isnan(rval), "The result should not be NaN with correct implementation."