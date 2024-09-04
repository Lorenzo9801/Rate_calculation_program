

from numpy import exp
from scipy import special
from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt 
from lmfit import Model, Parameters
import configparser
from npat import Isotope
from numpy import log as ln


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    config_dict = {}
    
    # Process 'paths' section
    for key in config.options('paths'):
        config_dict[key] = config.get('paths', key)

    # Process 'settings' section
    for key in config.options('settings'):
        config_dict[key] = config.get('settings', key)

    # Process 'costants' section
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



def PlotCrossSection(data, theorical_X, result):
    """
    Plot the experimental data and the fitted model curve for cross section. This function generates a plot comparing the experimental data to the fitted model
    curve.

    Parameters:
        data (numpy.ndarray): The experimental data, with the first column as the x-values and the second column as the y-values.
        theorical_X (numpy.ndarray): The x-values for plotting the theoretical model curve.
        result: The result object of the fit, containing the fit parameters.
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
    plt.draw()
    plt.show()
    

def PlotStoppingPower(data, theorical_X, result):
    """
    Plot the experimental data and the fitted model curve for stopping power.

    Parameters:
        data (numpy.ndarray): The experimental data, with the first column as the x-values and the second column as the y-values.
        theorical_X (numpy.ndarray): The x-values for plotting the theoretical model curve.
        result: The result object of the fit, containing the fit parameters.
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
    plt.draw()
    plt.show()




def Integral(n_slice, cs_params, sp_params, settings):
    """
    Calculate the total reaction rate by integrating over the projectile path.

    This function divides the projectile path into a specified number of slices, calculates
    the kinetic energy, cross-section, and stopping power for each slice, and integrates
    the reaction rate over the entire path.

    Parameters:
        n_slice (int): Number of slices to divide the projectile path into.
        cs_params (dict): Dictionary of parameters for the cross-section model function.
        sp_params (dict): Dictionary of parameters for the stopping power model function.
        config_file (str): Path to the configuration file containing additional parameters (e.g. atomic numbers, initial kinetic energy).

    Returns:
        The distance at which the projectile stops.
        The function prints the estimated reaction rate.
    """

    ZI = float(settings.get('zi')) #Atomic number
    AI = float(settings.get('ai')) #Mass number
    K_i = float(settings.get('k_i')) #Initial kinetic energy
    Ze = float(settings.get('ze')) #Charge state of the accelerated ion
    I0 = float(settings.get('i0')) #Initial beam current
    rhot = float(settings.get('rhot')) #Target density
    I=settings.get('i') #Isotope
    total_thickness=float(settings.get('total_thickness')) #Slice thickness in mm
    

    Iso=Isotope(I)
    PA=Iso.mass 
    #You multiply by 3600 because the library provides the half-life in hours and we need it in seconds
    HL=(Iso.half_life(Iso.optimum_units())*3600) #half life of the isotope
    decay_constant=ln(2)/(HL) 

    Mp= float(settings.get('mp'))  #Mass of a proton
    q_ele= float(settings.get('q_ele')) #Charge of an electtron
    cs= float(settings.get('cs')) #Speed of the light
    NA= float(settings.get('na')) #Avogadro number
    NT=0.01*rhot*NA*total_thickness/PA # Number of nuclei per unit area, where I multiply by 0.01 to express NT in nuclei per square millimeter (1/mm^2)

    slice_thickness = total_thickness / n_slice

    cumulative_energy_loss = 0
    rval = 0


    for k in range(n_slice): # This is the actual integral.
        energy_loss = stopping_power(slice_thickness*k, **sp_params)*slice_thickness # Calculate the energy lost in the k-th slice
        cumulative_energy_loss += energy_loss # Update the cumulative energy loss
    
        k_e_slice = K_i - cumulative_energy_loss  # Calculate the kinetic energy for this slice
        if k_e_slice < 0:
            k_e_slice = 0
            projectil_path= slice_thickness*k
            break

        sgm = cross_section(k_e_slice,**cs_params)*10**(-22)  # calculate the cross-section in mm².
        Itmp = I0 * np.sqrt(k_e_slice / K_i)  # Beam current in amperes (A) in the k-th slice.
        nptmp = Itmp / (Ze * q_ele)  # Number of particles in the beam in the k-th slice.
        rval += nptmp * NT * sgm   # Add reaction rate for this slice


  
    return rval, projectil_path