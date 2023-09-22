# This is a program that can measure and model the gravitational waves emitted by massive objects in space using general relativity, differential geometry, and signal processing.

# Import libraries
import numpy as np # for numerical computations
import scipy as sp # for scientific computing
import matplotlib.pyplot as plt # for plotting
import pycbc # for gravitational wave analysis
import emcee # for Markov chain Monte Carlo sampling
import corner # for plotting posterior distributions

# Define some constants
c = 3e8 # speed of light in m/s
G = 6.67e-11 # gravitational constant in m^3/kg/s^2
M_sun = 2e30 # mass of the sun in kg

# Define some parameters
f0 = 100 # initial frequency of the gravitational wave in Hz
M1 = 15 * M_sun # mass of the first object in kg
M2 = 10 * M_sun # mass of the second object in kg
s1x = 0.5 # spin component of the first object along the x-axis
s1y = 0.3 # spin component of the first object along the y-axis
s1z = 0.2 # spin component of the first object along the z-axis
s2x = -0.4 # spin component of the second object along the x-axis
s2y = -0.2 # spin component of the second object along the y-axis
s2z = -0.1 # spin component of the second object along the z-axis
e0 = 0.1 # initial eccentricity of the binary system
iota = np.pi / 3 # inclination angle of the binary system
phi0 = 0 # initial phase of the gravitational wave
ra = np.pi / 4 # right ascension of the source in radians
dec = np.pi / 6 # declination of the source in radians
psi = np.pi / 8 # polarization angle of the gravitational wave in radians
d = 1e9 # luminosity distance to the source in m

# Define some functions
def waveform_model(params):
    """This function generates the gravitational waveform model using pycbc library.
    
    Args:
        params (list): A list of model parameters, including the masses, spins, eccentricity, inclination, phase, right ascension, declination, polarization, and distance of the source.
    
    Returns:
        hp (pycbc.types.frequencyseries.FrequencySeries): The plus polarization of the waveform model.
        hc (pycbc.types.frequencyseries.FrequencySeries): The cross polarization of the waveform model.
    """
    M1, M2, s1x, s1y, s1z, s2x, s2y, s2z, e0, iota, phi0, ra, dec, psi, d = params # unpack the parameters
    
    # Define some options for the waveform model
    f_low = 20 # lower frequency cutoff in Hz
    f_high = None # higher frequency cutoff in Hz (None means use default value)
    delta_f = 1/4 # frequency resolution in Hz
    approximant = 'EccentricFD' # name of the waveform approximant
    
    # Generate the waveform model using pycbc.waveform.get_fd_waveform function
    hp, hc = pycbc.waveform.get_fd_waveform(approximant=approximant,
                                            mass1=M1,
                                            mass2=M2,
                                            spin1x=s1x,
                                            spin1y=s1y,
                                            spin1z=s1z,
                                            spin2x=s2x,
                                            spin2y=s2y,
                                            spin2z=s2z,
                                            eccentricity=e0,
                                            inclination=iota,
                                            coa_phase=phi0,
                                            distance=d,
                                            delta_f=delta_f,
                                            f_lower=f_low,
                                            f_final=f_high)
    
    return hp, hc

def detector_data(detector):
    """This function loads the detector data using pycbc library.
    
    Args:
        detector (str): The name of the detector, such as 'H1' or 'L1'.
    
    Returns:
        data (pycbc.types.timeseries.TimeSeries): The detector data as a time series.
    """
    start_time = 1126259446 # GPS start time of the data segment in seconds
    end_time = 1126259478 # GPS end time of the data segment in seconds
    
    # Load the detector data using pycbc.frame.query_and_read_frame function
    data = pycbc.frame.query_and_read_frame('GWOSC', detector + ':GWOSC-16KHZ_R1_STRAIN', start_time, end_time)
    
    return data

def detector_response(hp, hc, detector):
    """This function calculates the detector response to the gravitational wave using pycbc library.
    
    Args:
        hp (pycbc.types.frequencyseries.FrequencySeries): The plus polarization of the waveform model.
        hc (pycbc.types.frequencyseries.FrequencySeries): The cross polarization of the waveform model.
        detector (str): The name of the detector, such as 'H1' or 'L1'.
    
    Returns:
        ht (pycbc.types.multidetector.MultiDetComplexTimeSeries): The detector response as a complex time series.
    """
    
    # Get the detector object using pycbc.detector.Detector function
    det = pycbc.detector.Detector(detector)
    
    # Get the time delay between the source and the detector using det.time_delay_from_earth_center function
    time_delay = det.time_delay_from_earth_center(ra, dec, data.start_time)
    
    # Get the antenna pattern functions for plus and cross polarizations using det.antenna_pattern function
    Fp, Fc = det.antenna_pattern(ra, dec, psi, data.start_time)
    
    # Shift the waveform by the time delay and apply the antenna pattern functions using pycbc.waveform.apply_fseries_time_shift and pycbc.types.MultiDetComplexTimeSeries functions
    ht = pycbc.waveform.apply_fseries_time_shift(hp, time_delay) * Fp + pycbc.waveform.apply_fseries_time_shift(hc, time_delay) * Fc
    ht = pycbc.types.MultiDetComplexTimeSeries({detector: ht})
    
    return ht

def log_likelihood(params):
    """This function calculates the log likelihood of the data given the model using pycbc library.
    
    Args:
        params (list): A list of model parameters, including the masses, spins, eccentricity, inclination, phase, right ascension, declination, polarization, and distance of the source.
    
    Returns:
        logL (float): The log likelihood of the data given the model.
    """
    hp, hc = waveform_model(params) # generate the waveform model
    
    # Define a frequency domain filter using pycbc.filter.matched_filter function
    filt = pycbc.filter.matched_filter(hp, data, psd=psd, low_frequency_cutoff=f_low)
    
    # Calculate the log likelihood using pycbc.filter.sigma and pycbc.filter.snr functions
    sigma = np.sqrt(pycbc.filter.sigma(hp, psd=psd, low_frequency_cutoff=f_low))
    snr = pycbc.filter.snr(filt) / sigma
    logL = -0.5 * np.vdot(snr, snr).real
    
    return logL

def log_prior(params):
    """This function calculates the log prior of the model parameters using uniform distributions.
    
    Args:
        params (list): A list of model parameters, including the masses, spins, eccentricity, inclination, phase, right ascension, declination, polarization, and distance of the source.
    
    Returns:
        logP (float): The log prior of the model parameters.
    """
    M1, M2, s1x, s1y, s1z, s2x, s2y, s2z, e0, iota, phi0, ra, dec, psi, d = params # unpack the parameters
    
    # Define some bounds for the parameters
    M1_min = 5 * M_sun # minimum mass of the first object in kg
    M1_max = 50 * M_sun # maximum mass of the first object in kg
    M2_min = 5 * M_sun # minimum mass of the second object in kg
    M2_max = 50 * M_sun # maximum mass of the second object in kg
    s_min = -1 # minimum spin component
    s_max = 1 # maximum spin component
    e0_min = 0 # minimum eccentricity
    e0_max = 0.5 # maximum eccentricity
    iota_min = 0 # minimum inclination angle in radians
    iota_max = np.pi # maximum inclination angle in radians
    phi0_min = 0 # minimum phase in radians
    phi0_max = 2 * np.pi # maximum phase in radians
    ra_min = 0 # minimum right ascension in radians
    ra_max = 2 * np.pi # maximum right ascension in radians
    dec_min = -np.pi / 2 # minimum declination in radians
    dec_max = np.pi / 2 # maximum declination in radians
    psi_min = 0 # minimum polarization angle in radians
    psi_max = np.pi / 2 # maximum polarization angle in radians
    d_min = 1e8 # minimum distance in m
    d_max = 1e10 # maximum distance in m
    
    # Check if the parameters are within the bounds and return the log prior accordingly
    if (M1_min <= M1 <= M1_max) and (M2_min <= M2 <= M2_max) and (s_min <= s1x <= s_max) and (s_min <= s1y <= s_max) and (s_min <= s1z <= s_max) and (s_min <= s2x <= s_max) and (s_min <= s2y <= s_max) and (s_min <= s2z <= s_max) and (e0_min <= e0 <= e0_max) and (iota_min <= iota <= iota_max) and (phi0_min <= phi0 <= phi0_max) and (ra_min <= ra <= ra_max) and (dec_min <= dec <= dec_max) and (psi_min <= psi <= psi_max) and (d_min <= d <= d_max):
        logP = np.log(1 / (M1_max - M1_min)) + np.log(1 / (M2_max - M2_min)) + np.log(1 / (s_max - s_min))**6 + np.log(1 / (e0_max - e0_min)) + np.log(1 / (iota_max - iota_min)) + np.log(1 / (phi0_max - phi0_min)) + np.log(1 / (ra_max - ra_min)) + np.log(1 / (dec_max - dec_min)) + np.log(1 / (psi_max - psi_min)) + np.log(1 / (d_max - d_min))
        return logP
    else:
        return -np.inf

def log_posterior(params):
    """This function calculates the log posterior of the model parameters using Bayes' theorem.
    
    Args:
        params (list): A list of model parameters, including the masses, spins, eccentricity, inclination, phase, right ascension, declination, polarization, and distance of the source.
    
    Returns:
        logP (float): The log posterior of the model parameters.
    """
    logP = log_prior(params) # calculate the log prior
    
    if np.isfinite(logP): # check if the log prior is finite
        logL = log_likelihood(params) # calculate the log likelihood
        return logP + logL # return the log posterior
    else:
        return logP # return the log prior

# Load the detector data for two detectors: LIGO Hanford and LIGO Livingston
data_H1 = detector_data('H1')
data_L1 = detector_data('L1')

# Concatenate the data from both detectors into a single object using pycbc.types.MultiDetFrameData
data = pycbc.types.MultiDetFrameData({detector: data[detector] for detector in ['H1', 'L1']})

# Estimate the power spectral density of the data using pycbc.psd.welch function
psd = data.psd(4)

# Define the initial guess for the model parameters
params_init = [M1, M2, s1x, s1y, s1z, s2x, s2y, s2z, e0, iota, phi0, ra, dec, psi, d]

# Define the number of walkers and steps for the Markov chain Monte Carlo sampling
nwalkers = 32 # number of walkers
nsteps = 1000 # number of steps

# Initialize the sampler using emcee.EnsembleSampler function
sampler = emcee.EnsembleSampler(nwalkers, len(params_init), log_posterior)

# Run the sampler using sampler.run_mcmc function
state = sampler.run_mcmc(params_init, nsteps)

# Get the samples from the sampler using sampler.get_chain function
samples = sampler.get_chain(discard=100, thin=10, flat=True) # discard the first 100 steps as burn-in, thin by a factor of 10, and flatten the chain

# Plot the posterior distributions of the model parameters using corner.corner function
labels = ['M1', 'M2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'e0', 'iota', 'phi0', 'ra', 'dec', 'psi', 'd'] # labels for the parameters
fig = corner.corner(samples, labels=labels, show_titles=True) # plot the corner plot
plt.show()
