

import functions_definition
import numpy as np
import pytest
import hypothesis
from hypothesis import strategies as st
from hypothesis import settings
from hypothesis import given
import pandas as pd
import os
import configparser



# test if with such initial values for the cross section function I obtain a positive value
@given(x=st.just(10),xc=st.just(10),A=st.just(1),sigma=st.just(1),tau=st.just(1))
def test_cross_section(x,xc,A,sigma,tau):
    model= functions_definition.cross_section(x,xc,A,sigma,tau)
    assert model>0

# test if with such initial values for the stoppin power function I obtain a positive value
@given(x=st.just(10),xc=st.just(10),A=st.just(100),sigma=st.just(0.5),tau=st.just(2), C1=st.just(5))
def test_stopping_power(x,xc,A,sigma,tau, C1):
    model= functions_definition.stopping_power(x,xc,A,sigma,tau,C1)
    assert model>0




#test if the output of the fitting of the stopping power are complete
@given(A=st.just(100),sigma=st.just(0.5), tau=st.just(2), C1=st.just(5))
def test_output_fitting_sp(A,sigma,tau, C1):

    config = configparser.ConfigParser()
    config.read('configuration.txt')




    stopping_power = config.get('paths', 'stopping_power')
    stopping_power_data_file=np.loadtxt(os.path.expanduser(stopping_power),comments='%')
   
    theorical_x, theorical_y = functions_definition.Fitting("Stopping power", functions_definition.stopping_power, stopping_power_data_file, A, sigma, tau, C1)

    assert 'A'  in theorical_y.params
    assert 'xc' in theorical_y.params
    assert 'sigma' in theorical_y.params
    assert 'tau' in theorical_y.params
    assert 'C1' in theorical_y.params


#test if the output of the fitting of the cross section are complete
@given(A=st.just(10),sigma=st.just(0.5), tau=st.just(2),C1=st.just(5))
def test_output_fitting_cs(A,sigma,tau,C1):

    config = configparser.ConfigParser()
    config.read('configuration.txt')




    cross_section = config.get('paths', 'sezione_urto')

    cross_section = config.get('paths', 'sezione_urto')
    cross_section_data_file=np.loadtxt(os.path.expanduser(cross_section),comments='%')
    
    theorical_x, theorical_y = functions_definition.Fitting("Name", functions_definition.cross_section, cross_section_data_file, A, sigma, tau, C1)

    assert 'A'  in theorical_y.params
    assert 'xc' in theorical_y.params
    assert 'sigma' in theorical_y.params
    assert 'tau' in theorical_y.params
    assert 'C1' not in theorical_y.params
 






