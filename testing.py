
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


# Fitting tests


def test_fitting_with_linear_data():
    """
    Test the fitting function using a simple linear model y = m*x + b.
    
    GIVEN: A set of x-values and y-values generated from a known linear function.
    WHEN: The fitting function is applied with a linear model.
    THEN: The fitted parameters should closely match the original slope (m) and intercept (b).
    """

    def linear_model(x, m, b):
        """Simple linear model y = m*x + b."""
        return m * x + b

    true_m = 2.0
    true_b = 1.0
    x_data = np.linspace(0, 10, 100)
    y_data = linear_model(x_data, true_m, true_b)

    data = np.column_stack((x_data, y_data))

    init_params = {'m': 3.0, 'b': 0.0}

    theorical_x, result = Fitting(linear_model, data, **init_params)

    fitted_m = result.params['m'].value
    fitted_b = result.params['b'].value

    assert abs(fitted_m - true_m) < 1e-6, f"Expected slope {true_m}, but got {fitted_m}"
    assert abs(fitted_b - true_b) < 1e-6, f"Expected intercept {true_b}, but got {fitted_b}"


def test_fitting_with_parabolic_data():
    """
    Test the fitting function using a simple parabolic model y = a*x^2 + b*x + c.
    
    GIVEN: A set of x-values and y-values generated from a known parabolic function.
    WHEN: The fitting function is applied with a parabolic model.
    THEN: The fitted parameters should closely match the original coefficients a, b, and c.
    """

    def parabolic_model(x, a, b, c):
        """Simple parabolic model y = a*x^2 + b*x + c."""
        return a * x**2 + b * x + c

    true_a = 1.0
    true_b = -2.0
    true_c = 5.0
    x_data = np.linspace(-10, 10, 100)
    y_data = parabolic_model(x_data, true_a, true_b, true_c)

    data = np.column_stack((x_data, y_data))

    init_params = {'a': 0.5, 'b': 0.0, 'c': 0.0}
    
    theorical_x, result = Fitting(parabolic_model, data, **init_params)
    
    fitted_a = result.params['a'].value
    fitted_b = result.params['b'].value
    fitted_c = result.params['c'].value
    
    assert abs(fitted_a - true_a) < 1e-6, f"Expected a {true_a}, but got {fitted_a}"
    assert abs(fitted_b - true_b) < 1e-6, f"Expected b {true_b}, but got {fitted_b}"
    assert abs(fitted_c - true_c) < 1e-6, f"Expected c {true_c}, but got {fitted_c}"


def test_fitting_with_exponential_data():
    """
    Test the fitting function using a simple exponential model y = A * exp(k * x).
    
    GIVEN: A set of x-values and y-values generated from a known exponential function.
    WHEN: The fitting function is applied with an exponential model.
    THEN: The fitted parameters should closely match the original coefficients A and k.
    """
    
    # Define the exponential model within the test
    def exponential_model(x, A, k):
        """Simple exponential model y = A * exp(k * x)."""
        return A * np.exp(k * x)

    # Generate some perfect exponential data
    true_A = 2.0
    true_k = -0.5
    x_data = np.linspace(0, 10, 100)
    y_data = exponential_model(x_data, true_A, true_k)
    
    # Combine the data into a 2D array as required by the Fitting function
    data = np.column_stack((x_data, y_data))
    
    # Initial guesses for the parameters
    init_params = {'A': 1.0, 'k': -0.1}
    
    # Call the fitting function with the exponential model
    theorical_x, result = Fitting(exponential_model, data, **init_params)
    
    # Extract the fitted parameters
    fitted_A = result.params['A'].value
    fitted_k = result.params['k'].value
    
    # Assert that the fitted parameters are close to the true values
    assert abs(fitted_A - true_A) < 1e-6, f"Expected A {true_A}, but got {fitted_A}"
    assert abs(fitted_k - true_k) < 1e-6, f"Expected k {true_k}, but got {fitted_k}"

def test_fitting_with_linear_data_with_noise():
    """
    Test the fitting function using noisy linear data.
    
    GIVEN: A set of x-values and y-values generated from a linear function with added noise.
    WHEN: The fitting function is applied with a linear model.
    THEN: The fitted parameters should closely match the original slope and intercept, 
          despite the presence of noise.
    """
    
    # Define the linear model within the test
    def linear_model(x, a, b):
        """Linear model y = a * x + b."""
        return a * x + b

    # Generate some linear data with noise
    true_a = 2.0  # Slope
    true_b = 5.0  # Intercept
    x_data = np.linspace(0, 10, 100)
    
    # Generate y data with added Gaussian noise
    noise = np.random.normal(0, 0.5, size=x_data.shape)  # Standard deviation of noise is 0.5
    y_data = linear_model(x_data, true_a, true_b) + noise
    
    # Combine the data into a 2D array as required by the Fitting function
    data = np.column_stack((x_data, y_data))
    
    # Initial guesses for the parameters
    init_params = {'a': 1.0, 'b': 0.0}
    
    # Call the fitting function with the linear model
    theorical_x, result = Fitting(linear_model, data, **init_params)
    
    # Extract the fitted parameters
    fitted_a = result.params['a'].value
    fitted_b = result.params['b'].value
    
    # Assert that the fitted parameters are close to the true values within a reasonable tolerance
    assert abs(fitted_a - true_a) < 0.1, f"Expected slope {true_a}, but got {fitted_a}"
    assert abs(fitted_b - true_b) < 0.1, f"Expected intercept {true_b}, but got {fitted_b}"




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

