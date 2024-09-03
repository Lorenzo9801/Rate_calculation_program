
import numpy as np
import os
import configparser
import functions_definition

config_file = input("Please enter the name of the configuration file (default: configuration.txt): ")
if config_file == "":
    config_file = "configuration.txt"

config_dict = functions_definition.load_config(config_file)


cross_section = config_dict.get('sezione_urto')
stopping_power = config_dict.get('stopping_power')


#import data on cross-sections
cross_section_data=np.loadtxt(os.path.expanduser(cross_section),comments='%')
X_CS, result_CS=functions_definition.Fitting(functions_definition.cross_section, cross_section_data,A=10,sigma=1,tau=1)
functions_definition.Plotting(cross_section_data, functions_definition.cross_section, X_CS, result_CS)


params_cross_section = {
    'xc': result_CS.params['xc'].value,
    'A': result_CS.params['A'].value,
    'sigma': result_CS.params['sigma'].value,
    'tau': result_CS.params['tau'].value
}



#Importing data on stopping power.
stopping_power_data=np.loadtxt(os.path.expanduser(stopping_power),comments='%')
X_SP,result_SP=functions_definition.Fitting(functions_definition.stopping_power, stopping_power_data,A=150,sigma=0.5,tau=2,C1=5)
functions_definition.Plotting(stopping_power_data, functions_definition.stopping_power, X_SP, result_SP)



params_stopping_power = {
    'xc': result_SP.params['xc'].value,
    'A': result_SP.params['A'].value,
    'sigma': result_SP.params['sigma'].value,
    'tau': result_SP.params['tau'].value,
    'C1': result_SP.params['C1'].value
}

num_slice = int(config_dict.get('num_slice'))
functions_definition.Integral(num_slice,params_cross_section,params_stopping_power, config_dict)