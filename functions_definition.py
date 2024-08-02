

from numpy import exp
from scipy import special
from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt 
from lmfit import Model, Parameters
import pylab as py
import configparser
from npat import Isotope
from numpy import log as ln


def cross_section(x,xc,A,sigma,tau):
    """
    Calculate the cross-section value used for fitting experimental data.

    This function represents a convolution of a Gaussian and an exponential 
    function, often used in fitting processes to model the cross-section data.

    Parameters:
        x (float): The independent variable for which the cross-section is calculated.
        xc (float): The x-value at which the cross-section reaches its maximum.
        A (float): The amplitude of the Gaussian function.
        sigma (float): The standard deviation of the Gaussian function, which controls its width.
        tau (float): The decay constant of the exponential function.

    Returns:
        float: The calculated cross-section value for fitting.
    """
    
    y=0.5*(A/tau)*exp((0.5*(sigma/tau)**2-(x-xc)/tau))*(1+special.erf(((x-xc)/sigma-sigma/tau)/sqrt(2)))
    return y


def stopping_power(x,xc,A,sigma,tau,C1):
    """
    Calculate the stopping power value used for fitting experimental data.

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
        float: The calculated stopping power value for fitting.
    """
    y=0.5*(A/tau)*exp(0.5*(sigma/tau)**2+(x-xc)/tau)*(1+special.erf(((xc-x)/sigma-sigma/tau)/sqrt(2)))- np.piecewise(x, [x <= xc, x > xc], [C1, 0])
    return y


def Fitting(model_function,data,**init_params):
    """
    Fit a model to experimental data.

    This function applies a specified model to the experimental data. It starts with initial parameter values
    provided by the user and adjusts them to best match the model to the data.

    Parameters:
        model_function (function): The model function used for fitting (e.g., cross_section or stopping_power).
        data (numpy.ndarray): A 2D array where the first column is x-values and the second column is y-values.
        **init_params: Initial values for the model parameters.

    Returns:
            - theorical_x (numpy.ndarray): X-values for plotting the fitted model curve.
            - result (lmfit.model.ModelResult): The fit results, including parameter values.
  
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
    theorical_x = np.arange(min_x, max_x, (max_x - min_x) / 10000)
    # Fit the model to the data using the specified parameters
    result=model.fit(Y_Data,params=parameters,x=X_Data)

   
    return theorical_x,  result


# Make the plot of the data and of the fitting curve
def Plotting(data, model_function,theorical_X, result):
    """
    Plot the experimental data and the fitted model curve.

    This function generates a plot comparing the experimental data to the fitted model
    curve. It automatically adjusts the plot based on whether the model function is for
    stopping power or cross section.

    Parameters:
        data (numpy.ndarray): The experimental data, with the first column as the x-values and the second column as the y-values.
        model_function (function): The model function used for fitting (e.g. cross_section or stopping_power).
        theorical_X (numpy.ndarray): The x-values for plotting the theoretical model curve.
        result (lmfit.model.ModelResult): The result object containing the fit parameters.
    Raises:
        ValueError: If `model_function` is not either `stopping_power` or `cross_section`.
    """


    X_Data=data[:,0]
    Y_Data=data[:,1]

    Val_I1 = result.params['xc'].value
    Val_I2 = result.params['A'].value
    Val_I3 = result.params['sigma'].value
    Val_I4 = result.params['tau'].value
      
    fig1 = plt.figure(1,figsize=(10,6))
    
    plt.plot(X_Data,Y_Data,'b.')


    # If we are fitting the stopping power, we will need an additional parameter, C1.
    if model_function == stopping_power:
        Val_I5 =  result.params['C1'].value
        plt.plot(theorical_X,model_function(theorical_X,Val_I1,Val_I2,Val_I3,Val_I4, Val_I5))
        plt.title("Stopping Power")
        plt.xlabel("Distance (mm)")
        plt.ylabel("<dE/dx> (Mev/mm)")
        plt.draw()
        plt.show()  


    elif model_function == cross_section: 
        plt.plot(theorical_X,model_function(theorical_X,Val_I1,Val_I2,Val_I3,Val_I4))
        plt.title("Cross section")
        plt.xlabel("Kinetic Energy (MeV)")
        plt.ylabel("Cross Sectio (barn)")
        plt.draw()
        plt.show()  


    else:
        raise ValueError("model_function must be either 'stopping_power' or 'cross_section'")





# Calculate a numerical integral deviding the path of projectil into n_slice
# Use the fitting parameters to calculate the integral
def Integral(n_slice, cs_params, sp_params, config_file):
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
        The function prints the estimated reaction rate.
    """
    config = configparser.ConfigParser()

    config.read(config_file) # give the value of the parameters used to calcutate tthe integral
    ZI = float(config.get('settings','ZI')) #Atomic number
    AI = float(config.get('settings','AI')) #Mass number
    K_i = float(config.get('settings','K_i')) #initial kinetic energy
    Ze = float(config.get('settings','Ze')) #charge state of the accelerated ion
    I0 = float(config.get('settings','I0')) #initial beam current
    rhot = float(config.get('settings','rhot')) #target density
    I=config.get('settings', 'I') #isotope
    total_thickness=float(config.get('settings','total_thickness')) #slice thickness in mm
    

    Iso=Isotope(I)
    PA=Iso.mass 
    #You multiply by 3600 because the library provides the half-life in hours and we need it in seconds
    HL=(Iso.half_life(Iso.optimum_units())*3600) #half life of the isotope
    decay_constant=ln(2)/(HL) 

    Mp= float(config.get('costants','Mp'))  #mass of a proton
    q_ele= float(config.get('costants','q_ele')) #charge of an electtron
    cs= float(config.get('costants','cs')) #speed of the light
    NA= float(config.get('costants','NA')) #Avogadro number
    NT=0.001*rhot*NA*5/PA

    slice_thickness = total_thickness / n_slice

    cumulative_energy_loss = 0
    rval = 0


    for k in range(n_slice): # This is the actual integral.
        energy_loss = stopping_power(slice_thickness*k, **sp_params)*slice_thickness # Calculate the energy in the k-th slice
        cumulative_energy_loss += energy_loss # Update the cumulative energy loss
    
        k_e_slice = K_i - cumulative_energy_loss  # Calculate the kinetic energy for this slice
        if k_e_slice < 0:
            print(f"Projectile energy is exhausted at slice {k}.")
            k_e_slice = 0
            break
        sgm = cross_section(k_e_slice,**cs_params)*10**(-22)  # calculate the cross-section in mm².
        Itmp = I0 * np.sqrt(k_e_slice / K_i)  # Beam current in amperes (A) in the k-th slice.
        nptmp = Itmp / (Ze * (q_ele*10**6))  # Number of particles in the beam in the k-th slice.
        rval += nptmp * NT * sgm * slice_thickness  # Add reaction rate for this slice


    print("The value of the estimated rate is: ",rval," s^-1")
    return rval