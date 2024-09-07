
from functions_definition import cross_section, stopping_power, Fitting, Integral, calculate_initial_parameters, calculate_slice_params
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





def test_calculate_initial_parameters():
    """
    Test the calculate_initial_parameters function with a known set of parameters.
    """
    settings = {
        'zi': '6',
        'ai': '12',
        'k_i': '1000',
        'ze': '2',
        'i0': '1e-6',
        'rhot': '2.5',
        'i': '14C',  # Modificato per riflettere il formato corretto
        'total_thickness': '10',
        'mp': '1.6726219e-27',
        'q_ele': '1.602176634e-19',
        'cs': '299792458',
        'na': '6.02214076e23'

    }
    PA = 14.00324198843
    expected_NT =  0.01 * 2.5 * 6.02214076e23 * 10.0 / PA

    ZI, AI, K_i, Ze, I0, rhot, total_thickness, q_ele, Mp, cs, NT = calculate_initial_parameters(settings)



    assert ZI == 6.0, f"Expected 6.0 but got {ZI}"
    assert AI == 12.0, f"Expected 12.0 but got {AI}"
    assert K_i == 1000.0, f"Expected 1000.0 but got {K_i}"
    assert Ze == 2.0, f"Expected 2.0 but got {Ze}"
    assert I0 == 1e-6, f"Expected 1e-6 but got {I0}"
    assert rhot == 2.5, f"Expected 2.5 but got {rhot}"
    assert total_thickness == 10.0, f"Expected 10.0 but got {total_thickness}"
    assert q_ele == 1.602176634e-19, f"Expected 1.602176634e-19 but got {q_ele}"
    assert Mp == 1.6726219e-27, f"Expected 1.6726219e-27 but got {Mp}"
    assert cs == 299792458.0, f"Expected 299792458.0 but got {cs}"
    assert abs(NT - expected_NT) < 1e-10, f"Expected {expected_NT} but got {NT}"

    print("Test passed.")


def test_calculate_slice_params():
    """
    Test the calculate_slice_params function with known parameters and expected results.
    """
    # Parametri di input
    k_e_slice = 5000  
    cs_params = {'parameter1': 1, 'parameter2': 2} 
    sp_params = {'parameter1': 3, 'parameter2': 0.5}  
    K_i = 100
    I0 = 20
    Ze = 3  
    q_ele = 7

    mock_cross_section = 1.0e-02 
    mock_stopping_power = 2.0e-03

    expected_sigma = 10**(-24)
    expected_beam_current = 6.7353198
    expected_energy_loss = 2.0*10**(-3)
    with patch('functions_definition.cross_section', return_value=mock_cross_section), \
         patch('functions_definition.stopping_power', return_value=mock_stopping_power):
        

        sigma, beam_current, energy_loss = calculate_slice_params(k_e_slice, cs_params, sp_params, K_i, I0, Ze, q_ele)

   


    
        assert abs(sigma - expected_sigma) < 1e-3, f"Expected sigma: {expected_sigma}, but got {sigma}"
        assert abs(beam_current - expected_beam_current) < 1e-3, f"Expected beam_current: {expected_beam_current}, but got {beam_current}"
        assert abs(energy_loss - expected_energy_loss) < 1e-3, f"Expected energy_loss: {expected_energy_loss}, but got {energy_loss}"
    
    print('test passed')

