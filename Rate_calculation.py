
import numpy as np
import os
import configparser
import functions_definition

config_file = input("Please enter the name of the configuration file (default: configuration.txt): ")
if config_file == "":
    config_file = "configuration.txt"

config = configparser.ConfigParser()
config.read(config_file)




cross_section = config.get('paths', 'sezione_urto')
stopping_power = config.get('paths', 'stopping_power')


#import data on cross-sections
cross_section_data=np.loadtxt(os.path.expanduser(cross_section),comments='%')
X_CS, result_CS=functions_definition.Fitting(functions_definition.cross_section, cross_section_data,A=10,sigma=1,tau=1)
functions_definition.Plotting(cross_section_data, functions_definition.cross_section, X_CS, result_CS)
Val_I1 = result_CS.params['xc'].value
Val_I2 = result_CS.params['A'].value
Val_I3 = result_CS.params['sigma'].value
Val_I4 = result_CS.params['tau'].value


#Importing data on stopping power.
stopping_power_data=np.loadtxt(os.path.expanduser(stopping_power),comments='%')
X_SP,result_SP=functions_definition.Fitting(functions_definition.stopping_power, stopping_power_data,A=150,sigma=0.5,tau=2,C1=5)
functions_definition.Plotting(stopping_power_data, functions_definition.stopping_power, X_SP, result_SP)
Val_I5 = result_SP.params['xc'].value
Val_I6 = result_SP.params['A'].value
Val_I7 = result_SP.params['sigma'].value
Val_I8 = result_SP.params['tau'].value
Val_I9 = result_SP.params['C1'].value

params_cross_section = {
    'xc': Val_I1,
    'A': Val_I2,
    'sigma': Val_I3,
    'tau': Val_I4
}

params_stopping_power = {
    'xc': Val_I5,
    'A': Val_I6,
    'sigma': Val_I7,
    'tau': Val_I8,
    'C1': Val_I9
}

functions_definition.Integral(1000,params_cross_section,params_stopping_power, "configuration.txt")