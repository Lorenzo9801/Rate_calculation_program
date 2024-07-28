
# Given a target made of a certain material and a beam of projectile nuclei
# we want to measure the total reaction rate within the target
# The provided data apply to the reaction 64Ni(p,n)64Cu

# install the library
# pip install --user npat

# import a library containing nuclear physics data

from npat import Isotope 
import numpy as np
from numpy import log as ln
import os
from lmfit import Model
import matplotlib.pyplot as plt
import pylab as py
import configparser
import sys
from sys import argv
import functions_definition

config = configparser.ConfigParser()
config.read('configuration.txt')




cross_section = config.get('paths', 'sezione_urto')
stopping_power = config.get('paths', 'stopping_power')
print ( str(functions_definition.cross_section))


#import data on cross-sections
cross_section_data_file=np.loadtxt(os.path.expanduser(cross_section),comments='%')
X_CS,Y_CS=functions_definition.Fitting(functions_definition.cross_section, cross_section_data_file,A=10,sigma=1,tau=1)
functions_definition.Plotting("cross_section", cross_section_data_file, functions_definition.cross_section, X_CS, Y_CS)
Val_I1, Val_I2, Val_I3, Val_I4 = functions_definition.Plotting("cross_section", cross_section_data_file, functions_definition.cross_section, X_CS, Y_CS)


#Importing data on stopping power.
stopping_power_data_file=np.loadtxt(os.path.expanduser(stopping_power),comments='%')



X_SP,Y_SP=functions_definition.Fitting(functions_definition.stopping_power, stopping_power_data_file,A=150,sigma=0.5,tau=2,C1=5)
functions_definition.Plotting("Stopping power", stopping_power_data_file, functions_definition.stopping_power, X_SP, Y_SP)
Val_5, Val_I6, Val_I7, Val_I8, Val_I9 = functions_definition.Plotting("Stopping power", stopping_power_data_file, functions_definition.stopping_power, X_SP, Y_SP)


functions_definition.Integral(1000,Val_I1, Val_I2, Val_I3, Val_I4, Val_5, Val_I6, Val_I7, Val_I8, Val_I9)