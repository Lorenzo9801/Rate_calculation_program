

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

# fitting both the cross section and the stopping power
# IF YOU ARE FITTING THE STOPPING POWER USE AS title "Stopping power"
def Fitting(model_function,data_file,**init_params):

    X_Data=data_file[:,0]  #import the data from the data file
    Y_Data=data_file[:,1]

    model=Model(model_function) # use a theorical funcion as a model for the fitting
        
 # The term xc1 is the value of x corresponding to the maximum y
    initial_xc= X_Data[np.where(Y_Data == np.max(Y_Data))].item() 

    parameters = Parameters()
    parameters.add('xc', value=initial_xc)
    for param_name, init_value in init_params.items():
     parameters.add(param_name, value=init_value)


    min_x, max_x = X_Data.min(), X_Data.max()
    theorical_x = np.arange(min_x, max_x, (max_x - min_x) / 10000)
    theorical_y=model.fit(Y_Data,params=parameters,x=X_Data)

    return theorical_x,  theorical_y


# Make the plot of the data and of the fitting curve
def Plotting(title,data_file, theorical_function,theorical_X, result):


    X_Data=data_file[:,0]
    Y_Data=data_file[:,1]

    Val_I1 = result.params['xc'].value
    Val_I2 = result.params['A'].value
    Val_I3 = result.params['sigma'].value
    Val_I4 = result.params['tau'].value
      
    fig1 = plt.figure(1,figsize=(10,6))
    plt.title(title)
    plt.plot(X_Data,Y_Data,'b.')


    # If we are fitting the stopping power, we will need an additional parameter, C1.
    if title == "Stopping power":
        Val_I5 =  result.params['C1'].value
        plt.plot(theorical_X,theorical_function(theorical_X,Val_I1,Val_I2,Val_I3,Val_I4, Val_I5))
        plt.draw()
        plt.show()  
        return  Val_I1, Val_I2, Val_I3, Val_I4, Val_I5

    else: 
        plt.plot(theorical_X,theorical_function(theorical_X,Val_I1,Val_I2,Val_I3,Val_I4))
        plt.draw()
        plt.show()  
        return Val_I1, Val_I2, Val_I3, Val_I4






# Calculate a numerical integral deviding the path of projectil into n_slice
# Use the fitting parameters to calculate the integral
def Integral(n_slice, Val_I1, Val_I2, Val_I3, Val_I4, Val_I5, Val_I6, Val_I7, Val_I8, Val_I9):
    config = configparser.ConfigParser()

    config.read('configuration.txt') # give the value of the parameters used to calcutate tthe integral
    ZI = np.float(config.get('settings','ZI')) #Atomic number
    AI = np.float(config.get('settings','AI')) #Mass number
    K_i = np.float(config.get('settings','K_i')) #initial kinetic energy
    Ze = np.float(config.get('settings','Ze')) #charge state of the accelerated ion
    I0 = np.float(config.get('settings','I0')) #initial beam current
    rhot = np.float(config.get('settings','rhot')) #target density
    I=config.get('settings', 'I') #isotope
    slice=np.float(config.get('settings','slice')) #slice thickness in mm
    

    Iso=Isotope(I)
    PA=Iso.mass 
    #You multiply by 3600 because the library provides the half-life in hours and we need it in seconds
    HL=(Iso.half_life(Iso.optimum_units())*3600) #half life of the isotope
    decay_constant=ln(2)/(HL) 

    Mp= np.float(config.get('costants','Mp'))  #mass of a proton
    q_ele= np.float(config.get('costants','q_ele')) #charge of an electtron
    cs= np.float(config.get('costants','cs')) #speed of the light
    NA= np.float(config.get('costants','NA')) #Avogadro number
    NT=0.001*rhot*NA*5/PA

    rval = 0
    k_e_slice=0
    final_k_e = 0


    for k in range(n_slice): # This is the actual integral.
  
        k_e_slice = K_i - stopping_power(( slice * k ),Val_I5,Val_I6,Val_I7,Val_I8,Val_I9)   # Kinetic energy in MeV in the k-th slice.
        sgm = cross_section(k_e_slice,Val_I1,Val_I2,Val_I3,Val_I4)*10**(-22)  # calculate the cross-section in mm².
        Itmp = I0 * np.sqrt(k_e_slice / K_i)  # Beam current in amperes (A) in the k-th slice.
        nptmp = Itmp / (Ze * (q_ele*10**6))  # Number of particles in the beam in the k-th slice.
        rtmp = (nptmp * NT * sgm)  # I calculate the rate in the k-th slice.
        if k_e_slice <= final_k_e:
            rval= rtmp + rval
        final_k_e = k_e_slice



    print("The value of the estimated rate is: ",rval," s^-1")