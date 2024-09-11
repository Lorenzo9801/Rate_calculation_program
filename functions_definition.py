

from numpy import exp
from scipy import special
from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt 
from lmfit import Model, Parameters
import configparser
from npat import Isotope
from numpy import log as ln


def load_config(config_file="configuration.txt"):
    """
    Load configuration data from a text file and return it as a dictionary.

    The configuration file is divided into sections (e.g., 'paths', 'settings', 'constants'),
    and each section contains key-value pairs (e.g., 'ZI = 29').

    Parameters:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary with all the configuration settings, grouped by section.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    config_dict = {}
    
    for key in config.options('paths'):
        config_dict[key] = config.get('paths', key)

    for key in config.options('settings'):
        config_dict[key] = config.get('settings', key)

    for key in config.options('costants'):
        config_dict[key] = config.get('costants', key)

    return config_dict

def cross_section(x,xc,A,sigma,tau):
    """
    This function represents a convolution of a Gaussian and an exponential 
    function, often used in fitting processes to model the cross-section data.

    Parameters:
        x (float): The independent variable for which the cross-section is calculated.
        xc (float): The x-value at which the cross-section reaches its maximum.
        A (float): The amplitude of the Gaussian function.
        sigma (float): The standard deviation of the Gaussian function, which controls its width.
        tau (float): The decay constant of the exponential function.

    Returns:
        float: The calculated cross-section value.
    """
    
    exp_term = exp(0.5 * (sigma / tau) ** 2 - (x - xc) / tau)
    gaussian_term = 1 + special.erf(((x - xc) / sigma - sigma / tau) / sqrt(2))
    
    y = 0.5 * (A / tau) * exp_term * gaussian_term    

    return y


def stopping_power(x,xc,A,sigma,tau,C1):
    """
    This function represents a convolution of a Gaussian and an exponential 
    function, often used in fitting processes to model the stopping power data.

    Parameters:
        x (float): The independent variable for which the stopping power is calculated.
        xc (float): The x-value at which the stopping power reaches its maximum.
        A (float): The amplitude of the Gaussian function.
        sigma (float): The standard deviation of the Gaussian function, which controls its width.
        tau (float): The decay constant of the exponential function.
        C1 (float): Additional constant parameter.

    Returns:
        float: The calculated stopping power value.
    """
    exp_term = exp(0.5 * (sigma / tau) ** 2 + (x - xc) / tau)
    gaussian_term = 1 + special.erf(((xc - x) / sigma - sigma / tau) / sqrt(2))
    piecewise_term = np.piecewise(x, [x <= xc, x > xc], [C1, 0])

    y = 0.5 * (A / tau) * exp_term * gaussian_term - piecewise_term

    return y


def Fitting(model_function,data,num_points=10000,**init_params):
    """
    Fit a model to experimental data.

    This function applies a specified model to the experimental data. It starts with initial parameter values
    provided by the user and adjusts them to best match the model to the data.

    Parameters:
        model_function (function): The model function used for fitting (cross_section or stopping_power).
        data (numpy.ndarray): A 2D array where the first column is x-values and the second column is y-values.
        **init_params: Initial values for the model parameters.

    Returns:
            - theorical_x (numpy.ndarray): X-values for plotting the fitted model curve.
            - result: The fit results, including parameter values.
  
    Raises:
        ValueError: If the data is not a 2D numpy array with two columns.

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data should be a 2D numpy array with two columns (x and y values).")
    
    X_Data=data[:,0]  #import the data from the data file
    Y_Data=data[:,1]

    model=Model(model_function) # use a model_function as a model for the fitting
        
    # The term xc1 is the value of x corresponding to the maximum y
    initial_xc= X_Data[np.where(Y_Data == np.max(Y_Data))].item() 

    parameters = Parameters()
    parameters.add('xc', value=initial_xc)
    for param_name, init_value in init_params.items():
        parameters.add(param_name, value=init_value)

    # Determine the range of x-values for generating the theoretical curve
    min_x, max_x = X_Data.min(), X_Data.max()
    # Generate x-values over the range [min_x, max_x] with a high resolution for plotting
    theorical_x = np.arange(min_x, max_x, (max_x - min_x) / num_points)
    # Fit the model to the data using the specified parameters
    result=model.fit(Y_Data,params=parameters,x=X_Data)

   
    return theorical_x,  result



def PlotCrossSection(data, theorical_X, result, show_plot=True, save_plot=True, filename='cross_section_plot.png'):
    """
    Plot the experimental data and the fitted model curve for cross section. This function generates a plot comparing the experimental data to the fitted model
    curve.  The plot can be displayed and/or saved to a file.

    Parameters:
        data (numpy.ndarray): The experimental data, with the first column as the x-values and the second column as the y-values.
        theorical_X (numpy.ndarray): The x-values for plotting the theoretical model curve.
        result: The result object of the fit, containing the fit parameters.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        save_plot (bool, optional): Whether to save the plot as an image file. Defaults to True.
        filename (str, optional): The name of the file to save the plot if `save_plot` is True. Defaults to 'cross_section_plot.png'.

    """
    X_Data = data[:,0]
    Y_Data = data[:,1]

    Val_I1 = result.params['xc'].value
    Val_I2 = result.params['A'].value
    Val_I3 = result.params['sigma'].value
    Val_I4 = result.params['tau'].value

    plt.figure(figsize=(10,6))
    plt.plot(X_Data, Y_Data, 'b.')
    plt.plot(theorical_X, cross_section(theorical_X, Val_I1, Val_I2, Val_I3, Val_I4))
    plt.title("Cross Section")
    plt.xlabel("Kinetic Energy (MeV)")
    plt.ylabel("Cross Section (barn)")

    if save_plot:
        plt.savefig(filename)
        print(f"Cross section plot has been saved as '{filename}'")
  
    if show_plot:
        plt.show() 
    
    plt.close()  


    

def PlotStoppingPower(data, theorical_X, result, show_plot=True, save_plot=True, filename='stopping_power_plot.png'):
    """
    Plot the experimental data and the fitted model curve for stopping power. This function generates a plot comparing the experimental data to the fitted model
    curve. The plot can be displayed and/or saved to a file.

    Parameters:
        data (numpy.ndarray): The experimental data, with the first column as the x-values and the second column as the y-values.
        theorical_X (numpy.ndarray): The x-values for plotting the theoretical model curve.
        result: The result object of the fit, containing the fit parameters.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        save_plot (bool, optional): Whether to save the plot as an image file. Defaults to True.
        filename (str, optional): The name of the file to save the plot if `save_plot` is True. Defaults to 'stopping_power_plot.png'.

    Returns:
    """
    X_Data = data[:,0]
    Y_Data = data[:,1]

    Val_I1 = result.params['xc'].value
    Val_I2 = result.params['A'].value
    Val_I3 = result.params['sigma'].value
    Val_I4 = result.params['tau'].value
    Val_I5 = result.params['C1'].value

    plt.figure(figsize=(10,6))
    plt.plot(X_Data, Y_Data, 'b.')
    plt.plot(theorical_X, stopping_power(theorical_X, Val_I1, Val_I2, Val_I3, Val_I4, Val_I5))
    plt.title("Stopping Power")
    plt.xlabel("Distance (mm)")
    plt.ylabel("<dE/dx> (Mev/mm)")

    if save_plot:
        plt.savefig(filename)
        print(f"Stopping power plot has been saved as '{filename}'")
  
    if show_plot:
        plt.show() 
    
    plt.close()  





def calculate_initial_parameters(settings):
    """
    Calculate and return the initial parameters needed for integration.

    Parameters:
        settings (dict): Dictionary of required parameters.

    Returns:
        dict: A dictionary containing the calculated parameters.
    """
    ZI = float(settings.get('zi'))
    AI = float(settings.get('ai'))
    K_i = float(settings.get('k_i'))
    Ze = float(settings.get('ze'))
    I0 = float(settings.get('i0'))
    rhot = float(settings.get('rhot'))
    I = settings.get('i')
    total_thickness = float(settings.get('total_thickness'))
    Mp = float(settings.get('mp'))
    q_ele = float(settings.get('q_ele'))
    cs = float(settings.get('cs'))
    NA = float(settings.get('na'))

    Iso = Isotope(I)
    PA = Iso.mass
    HL = Iso.half_life(Iso.optimum_units()) * 3600  # Half-life in seconds
    decay_constant = np.log(2) / HL  # Decay constant

    NT = 0.01 * rhot * NA * total_thickness / PA  # Number of nuclei per unit area (1/mm^2)
    
    return ZI, AI, K_i, Ze, I0, rhot, total_thickness, q_ele, Mp, cs, NT

def calculate_slice_params(k_e_slice, cs_params, sp_params, K_i, I0, Ze, q_ele):
    """
    Calculate the cross-section and beam current for a given kinetic energy slice.

    Parameters:
        k_e_slice (float): Kinetic energy in the current slice.
        cs_params (dict): Parameters for the cross-section model.
        sp_params (dict): Parameters for the stopping power model.
        K_i (float): Initial kinetic energy.
        I0 (float): Initial beam current.
        Ze (float): Charge state of the ion.
        q_ele (float): Charge of an electron.

    Returns:
        tuple: Cross-section and beam current for the given slice.
    """
    sigma = cross_section(k_e_slice, **cs_params) * 10**(-22)  # Convert to mm²

    Itmp = I0 * np.sqrt(k_e_slice / K_i)
    beam_current = Itmp / (Ze * q_ele)
    

    
    return sigma, beam_current


def integrate_slice(slice_thickness, k, cumulative_energy_loss, cs_params, sp_params, K_i, I0, Ze, q_ele, NT):
    """
    Integrate over a single slice of the projectile path.

    Parameters:
        slice_thickness (float): Thickness of the slice.
        k (int): Current slice index.
        cumulative_energy_loss (float): The cumulative energy loss so far.
        cs_params (dict): Parameters for the cross-section model.
        sp_params (dict): Parameters for the stopping power model.
        K_i (float): Initial kinetic energy.
        I0 (float): Initial beam current.
        Ze (float): Charge state of the ion.
        q_ele (float): Charge of an electron.
        NT (float): Number of nuclei per unit area (1/mm²).

    Returns:
        tuple: Reaction rate contribution for this slice and the updated cumulative energy loss.
    """

    energy_loss = stopping_power(slice_thickness * k, **sp_params) * slice_thickness
    cumulative_energy_loss += energy_loss

    k_e_slice = K_i - cumulative_energy_loss
    if k_e_slice < 0:
        k_e_slice = 0
        return 0, cumulative_energy_loss 

    sigma, beam_current = calculate_slice_params(k_e_slice, cs_params, sp_params, K_i, I0, Ze, q_ele)
    reaction_rate = beam_current * NT * sigma

    return reaction_rate, cumulative_energy_loss


def Integral(n_slice, cs_params, sp_params, settings):
    """
    Calculate the total reaction rate by integrating over the projectile path.

    This function divides the projectile path into a specified number of slices and integrates
    the reaction rate over the entire path.

    Parameters:
        n_slice (int): Number of slices to divide the projectile path into.
        cs_params (dict): Dictionary of parameters for the cross-section model function.
        sp_params (dict): Dictionary of parameters for the stopping power model function.
        settings (dict): Dictionary of additional parameters required for the calculation.

    Returns:
        tuple: The total reaction rate and the distance at which the projectile stops.
    """
    # Get initial parameters
    ZI, AI, K_i, Ze, I0, rhot, total_thickness, q_ele, Mp, cs, NT = calculate_initial_parameters(settings)

    # Initialize variables
    slice_thickness = total_thickness / n_slice
    
    cumulative_energy_loss=0
    rval = 0
    projectile_path = total_thickness  # Default value if the projectile does not stop

    for k in range(n_slice):
        reaction_rate, cumulative_energy_loss = integrate_slice(slice_thickness, k, cumulative_energy_loss, cs_params, sp_params, K_i, I0, Ze, q_ele, NT)
        rval += reaction_rate
        if cumulative_energy_loss >= K_i:
            projectile_path = slice_thickness * k
            break

    return rval, projectile_path
