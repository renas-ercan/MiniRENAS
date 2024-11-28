import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def generateexp_refl(wavelengths, x, y, wave_start, wave_end):
    """
    Generates the experimental reflectance.
    """
    # Create a 2D array with wavelengths, reference, and spectrum data
    vector = np.column_stack((wavelengths, x, y))
    
    # Interpolate the data in the wavelength range 650 to 915
    waves = np.arange(650, 916)
    interp_func = interp1d(wavelengths, vector, axis=0, fill_value="extrapolate")
    interp_data = interp_func(waves)
    
    # Find indices corresponding to the wave_start and wave_end
    p = np.where(np.round(interp_data[:, 0]) == wave_start)[0][0]
    q = np.where(np.round(interp_data[:, 0]) == wave_end)[0][0]
    
    # Adjust q to include wave_end
    q = q + 1  # Include wave_end index

    # Extract the reference and spectrum data between the selected wavelength range
    r = interp_data[p:q, 1]
    s = interp_data[p:q, 2]

    # Calculate the experimental reflectance
    refl = r / (s * 100000)
    
    # Smooth the reflectance using Savitzky-Golay filter for smoothing
    exp_refl = savgol_filter(refl, window_length=11, polyorder=3)

    # Calculate the 1st derivative of the experimental reflectance
    diff_1_exp_refl = np.diff(exp_refl)
    
    # Calculate the 2nd derivative of the experimental reflectance
    diff_2_exp_refl = np.diff(diff_1_exp_refl)
    
    return exp_refl, diff_1_exp_refl, diff_2_exp_refl
