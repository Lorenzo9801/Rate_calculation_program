
import numpy as np
import os
import configparser
import functions_definition
import matplotlib.pyplot as plt


config_file = input("Please enter the name of the configuration file (default: configuration.txt): ")
if config_file == "":
    config_file = "configuration.txt"

config_dict = functions_definition.load_config(config_file)

cross_section = config_dict.get('sezione_urto')
stopping_power = config_dict.get('stopping_power')


show_plots = config_dict.get('show_plots')
save_plots = config_dict.get('save_plots')
save_results = config_dict.get('save_results')

#import data on cross-sections
cross_section_data=np.loadtxt(os.path.expanduser(cross_section),comments='%')
X_CS, result_CS=functions_definition.Fitting(functions_definition.cross_section, cross_section_data,A=10,sigma=1,tau=1)

if show_plots == 'yes':
    functions_definition.PlotCrossSection(cross_section_data, X_CS, result_CS)
if save_plots == 'yes':
    plt.savefig('cross_section_plot.png')
    print("Cross section plot has been saved as 'cross_section_plot.png'")
    plt.close()

params_cross_section = {
    'xc': result_CS.params['xc'].value,
    'A': result_CS.params['A'].value,
    'sigma': result_CS.params['sigma'].value,
    'tau': result_CS.params['tau'].value
}



#Importing data on stopping power.
stopping_power_data=np.loadtxt(os.path.expanduser(stopping_power),comments='%')
X_SP,result_SP=functions_definition.Fitting(functions_definition.stopping_power, stopping_power_data,A=150,sigma=0.5,tau=2,C1=5)

if show_plots == 'yes':
    functions_definition.PlotStoppingPower(stopping_power_data, X_SP, result_SP)
if save_plots == 'yes':
    plt.savefig('stopping_power_plot.png')
    print("Stopping power plot has been saved as 'stopping_power_plot.png'")
    plt.close()

params_stopping_power = {
    'xc': result_SP.params['xc'].value,
    'A': result_SP.params['A'].value,
    'sigma': result_SP.params['sigma'].value,
    'tau': result_SP.params['tau'].value,
    'C1': result_SP.params['C1'].value
}

num_slice = int(config_dict.get('num_slice'))

rval,projectil_path=functions_definition.Integral(num_slice,params_cross_section,params_stopping_power, config_dict)

if config_dict.get('save_results').lower() == 'yes':
    result_file = config_dict.get('result_file', 'result.txt')
    with open(result_file, "w") as file:
        file.write(f"The value of the estimated rate is: {rval} s^-1\n")
        file.write(f"The path of the projectile is {projectil_path} mm.\n")

    print(f"Results have been saved to {result_file}.")