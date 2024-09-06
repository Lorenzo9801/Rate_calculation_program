
from functions_definition import cross_section, stopping_power, Fitting, Integral
import numpy as np
import pytest
from unittest.mock import patch

# cross_section function tests

def test_cross_section_tau():
    """
    Test the `cross_section` function's behavior with varying tau values.

    GIVEN: The cross-section function with parameters xc, A, sigma, and different values for tau.
    WHEN: The cross-section function is called with x equal to xc and varying tau values.
    THEN: The cross-section should decrease as tau increases, meaning the results should be ordered such that
          the cross section for a smaller tau is greater than that for a larger tau.
    """
    xc = 5.0
    A = 10
    sigma = 1
    
    tau_values = [0.5, 1, 2]
    x = xc 
    
    results = [cross_section(x, xc, A, sigma, tau) for tau in tau_values]
    
    assert results[0] > results[1] > results[2], "The cross section should increase with increasing tau"

def test_cross_section_sigma():
    """
    Test the `cross_section` function's behavior with varying sigma values.

    GIVEN: The cross-section function with parameters xc, A, tau, and different values for sigma.
    WHEN: The cross-section function is called with x equal to xc and varying sigma values.
    THEN: The cross-section should decrease as sigma increases, meaning the results should be ordered such that
          the cross section for a smaller sigma is greater than that for a larger sigma.
    """
    xc = 5.0
    A = 10
    tau = 2
    
    sigma_values = [0.5, 1, 2]
    x = xc 
    
    results = [cross_section(x, xc, A, sigma, tau) for sigma in sigma_values]
    
    assert results[0] > results[1] > results[2], "The cross section should decrease as sigma increases"


def test_cross_section_amplitude():
    """
    Test the `cross_section` function's behavior with varying amplitude values.

    GIVEN: The cross-section function with parameters xc, sigma, tau, and different values for amplitude A.
    WHEN: The cross-section function is called with x equal to xc and varying amplitude values.
    THEN: The cross-section should increase as amplitude A increases, meaning the results should be ordered such that
          the cross section for a smaller amplitude is less than that for a larger amplitude.
    """

    xc = 5.0
    sigma = 1
    tau = 2
    

    A_values = [5, 10, 20]
    x = xc  
    
    results = [cross_section(x, xc, A, sigma, tau) for A in A_values]
    
   
    assert results[0] < results[1] < results[2], "The cross section should increase with increasing amplitude A"

def test_cross_section_around_xc():
    """
    Test the `cross_section` function's behavior around the central value xc.

    GIVEN: The cross-section function with parameters xc, A, sigma, tau, and a range of x values around xc.
    WHEN: The cross-section function is called with varying x values close to xc.
    THEN: The maximum cross-section value should be near xc, meaning the maximum should be within 1.5 units of xc.
    """
    xc = 5.0
    A = 10
    sigma = 1
    tau = 2


    x_values = np.linspace(xc - 2, xc + 2, 1000)

    cross_section_values = [cross_section(x, xc, A, sigma, tau) for x in x_values]

    x_max_index = np.argmax(cross_section_values)
    x_max = x_values[x_max_index]

    assert abs(x_max - xc) < 1.5, f"Maximum should be near xc. Found maximum at {x_max}, expected around {xc}"


def test_cross_section_behavior_at_limits():
    """
    Test the `cross_section` function's behavior for large x values.

    GIVEN: The cross-section function with parameters xc, A, sigma, tau, and a very large x value.
    WHEN: The cross-section function is called with a large x value.
    THEN: The cross-section should be near zero for large x values, meaning the result should be less than 1e-6.
    """

    xc = 5.0
    A = 10
    sigma = 1
    tau = 2
    
    x_large = 1000
    
    result_large = cross_section(x_large, xc, A, sigma, tau)

    
    assert result_large < 1e-6, "Cross section should be near zero for large x"


# stopping_power function tests


def test_stopping_power_at_infinity():
    """
    Test that the stopping power approaches 0 as x becomes very large.
    
    GIVEN: A large value of x and known parameters xc, A, sigma, tau, and C1.
    WHEN: The stopping power is calculated for this large x-value.
    THEN: The function should approach 0 for large x.
    """
    x=1000
    xc = 5.0
    A = 10
    sigma = 1
    tau = 2
    C1 = 0.5
    
    result = stopping_power(x, xc, A, sigma, tau, C1)
    
    assert result < 1e-6, f"Stopping power at x=1000 should be near zero, got {result}"

def test_stopping_power_with_different_A():
    """
    Test the effect of changing A on the stopping power.
    
    GIVEN: Different values of A and fixed values for xc, sigma, tau, and C1.
    WHEN: The stopping power is calculated for x close to xc.
    THEN: The stopping power should increase as A increases.
    """
    xc = 5.0
    sigma = 1
    tau = 2
    C1 = 0.5
    
    A_values = [5, 10, 20]
    x = xc
    
    results = [stopping_power(x, xc, A, sigma, tau, C1) for A in A_values]
    
    assert results[0] < results[1] < results[2], f"Stopping power should increase with A, got {results}"

def test_stopping_power_with_different_sigma():
    """
    Test the effect of changing sigma on the stopping power.
    
    GIVEN: Different values of sigma and fixed values for xc, A, tau, and C1.
    WHEN: The stopping power is calculated for x close to xc.
    THEN: The stopping power should decrease as sigma increases, as a larger sigma broadens the Gaussian function.
    """
    xc = 5.0
    A = 10
    tau = 2
    C1 = 0.5
    
    sigma_values = [0.5, 1, 2]  # Smaller sigma -> narrower Gaussian; larger sigma -> broader Gaussian
    x = xc
    
    results = [stopping_power(x, xc, A, sigma, tau, C1) for sigma in sigma_values]
    
    # Check that the stopping power decreases with increasing sigma
    assert results[0] > results[1] > results[2], f"Stopping power should decrease with increasing sigma, got {results}"


def test_fitting_invalid_data():
    """
    Test the `Fitting` function with data that is not in the required format.

    GIVEN: A 1D numpy array as input data, which should be a 2D array with two columns.
    WHEN: The `Fitting` function is called with this 1D array as the input data.
    THEN: The `Fitting` function should raise a ValueError indicating that the data should be a 2D numpy array with exactly two columns.
    """
    def model_function(x, xc):
        return np.exp(-((x - xc)**2))

    invalid_data = np.array([1, 0.5, 2, 0.7, 3, 0.2])  
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
    THEN: The estimated parameters `xc`, `A`, `sigma` and `tau` should all be positive or non-negative.
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



# def test_integral_invalid_energy_loss():
#     """
#     Test the `Integral` function when encountering extreme values for energy loss.

#     GIVEN: Mock implementations of `cross_section` and `stopping_power`, where `stopping_power` simulates a very high energy loss.
#     WHEN: The `Integral` function is called using these mock implementations.
#     THEN: The function should not return NaN, ensuring that it handles extreme energy loss values properly and does not produce invalid results.
#     """
#     def mock_cross_section(energy, **params):
#         return 1.0  # Simulates a constant cross-section function
    
#     def mock_stopping_power(distance, **params):
#         return 100.0  # Simulates a very high energy loss

#     # Example parameters
#     cs_params = {}
#     sp_params = {}
#     config = 'configuration.txt'



#     with patch('functions_definition.cross_section', mock_cross_section), \
#          patch('functions_definition.stopping_power', mock_stopping_power):
        
#         # Execute the test
#         rval = Integral(10, cs_params, sp_params, config)
        
#         # Verify that the result is not NaN
#         assert not np.isnan(rval), "The result should not be NaN with correct implementation."

