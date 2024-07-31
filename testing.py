
from functions_definition import cross_section, stopping_power, Fitting
import numpy as np
import pytest



def test_fitting_cs_params_positive():

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

    x_data = np.linspace(0, 10, 100)
    y_data = stopping_power(x_data, xc=5, A=150, sigma=0.5, tau=2, C1=5) 
    data = np.column_stack((x_data, y_data))
    
    theorical_x, result = Fitting(stopping_power, data, A=150, sigma=0.5, tau=2, C1=5)
    
    params = result.params
    
    assert params['xc'].value > 0, "Parameter 'xc' should be positive"
    assert params['A'].value > 0, "Parameter 'A' should be positive"
    assert params['sigma'].value > 0, "Parameter 'sigma' should be positive"
    assert params['tau'].value > 0, "Parameter 'tau' should be positive"