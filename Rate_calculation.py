
# Dato un target di un un certo materiale e un fascio di nuclei proiettili 
#si vuole misurare il rate totale di reazione all'interno del target
#I dati riportati valgono per la reazione 64Ni(p,n)64Cu

#installare libreria
#pip install --user npat

#si importa una libreria contenente dati di fisica nucleare

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
X_CS,Y_CS=functions_definition.Fitting("Cross Section",functions_definition.cross_section, cross_section_data_file,10,1,1,1)
functions_definition.Plotting("cross_section", cross_section_data_file, functions_definition.cross_section, X_CS, Y_CS)
Val_I1, Val_I2, Val_I3, Val_I4 = functions_definition.Plotting("cross_section", cross_section_data_file, functions_definition.cross_section, X_CS, Y_CS)


#Importing data on stopping power.
stopping_power_data_file=np.loadtxt(os.path.expanduser(stopping_power),comments='%')



X_SP,Y_SP=functions_definition.Fitting("Stopping power",functions_definition.stopping_power, stopping_power_data_file,150,0.5,2,5)
functions_definition.Plotting("Stopping power", stopping_power_data_file, functions_definition.stopping_power, X_SP, Y_SP)
Val_5, Val_I6, Val_I7, Val_I8, Val_I9 = functions_definition.Plotting("Stopping power", stopping_power_data_file, functions_definition.stopping_power, X_SP, Y_SP)


functions_definition.Integral(1000,Val_I1, Val_I2, Val_I3, Val_I4, Val_5, Val_I6, Val_I7, Val_I8, Val_I9)