import pytest
import numpy as np
from lmfit import Model, Parameters
from functions_definition import Fitting, cross_section, stopping_power, Integral
import configparser
from npat import Isotope
from unittest.mock import patch

# def test_fitting_no_error():
#     # Genera dati di esempio
#     x_data = np.linspace(0, 10, 100)
#     y_data = cross_section(x_data, xc=5, A=10, sigma=1, tau=2) + np.random.normal(0, 0.1, x_data.size)
#     data = np.column_stack((x_data, y_data))
    
#     try:
#         theorical_x, result = Fitting(cross_section, data, A=10, sigma=1, tau=2)
#     except Exception as e:
#         pytest.fail(f"Fitting failed with exception: {e}")

#     # Verifica che il fitting abbia successo
#     assert result.success, "The fitting did not succeed"



# def test_fitting_params():
#     # Dati di esempio
#     x_data = np.linspace(0, 10, 100)
#     y_data = cross_section(x_data, xc=5, A=10, sigma=1, tau=2) + np.random.normal(0, 0.1, x_data.size)
#     data = np.column_stack((x_data, y_data))
    
#     theorical_x, result = Fitting(cross_section, data, A=10, sigma=1, tau=2)
    
#     params = result.params
#     assert 'xc' in params, "Parameter 'xc' should be present in the results"
#     assert 'A' in params, "Parameter 'A' should be present in the results"
#     assert 'sigma' in params, "Parameter 'sigma' should be present in the results"
#     assert 'tau' in params, "Parameter 'tau' should be present in the results"



def test_fitting_cs_params_positive():

    x_data = np.linspace(0, 10, 100)
    y_data = cross_section(x_data, xc=5, A=10, sigma=1, tau=2) + np.random.normal(0, 0.1, x_data.size)
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(cross_section, data, A=10, sigma=1, tau=2)
    
    params = result.params
    
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"

def test_fitting_cs_params_positive():
    # Dati di esempio
    x_data = np.linspace(0, 10, 100)
    y_data = stopping_power(x_data, xc=5, A=150, sigma=0.5, tau=2, C1=5) + np.random.normal(0, 0.1, x_data.size)
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(stopping_power, data, A=150, sigma=0.5, tau=2, C1=5)
    
    params = result.params
    
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"
 


def test_fitting_convergence():
    # Genera dati di esempio
    x_data = np.linspace(0, 10, 100)
    y_data = cross_section(x_data, xc=5, A=10, sigma=1, tau=2) + np.random.normal(0, 0.1, x_data.size)
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(cross_section, data, A=10, sigma=1, tau=2)
    
    # Verifica che il fitting sia riuscito
    assert result.success, "The fitting did not converge successfully"

def test_fitting_stopping_power_convergence():
    # Dati di esempio
    x_data = np.linspace(0, 10, 100)
    y_data = stopping_power(x_data, xc=5, A=150, sigma=0.5, tau=2, C1=5) + np.random.normal(0, 0.1, x_data.size)
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(stopping_power, data, A=150, sigma=0.5, tau=2, C1=5)
    
    # Verifica che il fitting sia riuscito
    assert result.success, "The fitting did not converge successfully"


def test_fitting_robustness():
    # Genera dati variabili con rumore
    x_data = np.linspace(0, 10, 100)
    true_params = {'xc': 5, 'A': 10, 'sigma': 1, 'tau': 2}
    y_data = cross_section(x_data, **true_params) + np.random.normal(0, 0.5, x_data.size)
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(cross_section, data, **true_params)
    
    # Verifica che il fitting sia riuscito e i parametri siano positivi
    assert result.success, "The fitting did not converge successfully"
    
    params = result.params
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"





def test_fitting_invalid_data():
    def model_function(x, xc):
        return np.exp(-((x - xc)**2))

    invalid_data = np.array([1, 0.5, 2, 0.7, 3, 0.2])  # Non è un array 2D
    init_params = {'xc': 2}
    
    with pytest.raises(ValueError, match="Data should be a 2D numpy array with two columns"):
        Fitting(model_function, invalid_data, **init_params)



import pytest
import numpy as np

def test_integral_invalid_energy_loss():
    def mock_cross_section(energy, **params):
        return 1.0  # Simula una funzione di cross-section costante
    
    def mock_stopping_power(distance, **params):
        return 100.0  # Simula una perdita di energia molto alta

    # Parametri di esempio
    cs_params = {}
    sp_params = {}
    config = 'configuration.txt'



    with patch('functions_definition.cross_section', mock_cross_section), \
         patch('functions_definition.stopping_power', mock_stopping_power):
        
        # Esegui il test
        rval = Integral(10, cs_params, sp_params, config)
        
        # Verifica che il risultato non sia NaN
        assert not np.isnan(rval), "The result should not be NaN with correct implementation."