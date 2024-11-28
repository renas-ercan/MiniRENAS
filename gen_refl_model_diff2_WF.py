import numpy as np
from scipy.signal import savgol_filter
from load_extinction_coeff import load_extinction_coeff

def gen_refl_model_diff2_WF(parameters, xdata):
    """
    Generates the reflectance model for given wavelengths and parameters.
    """
    coeff = load_extinction_coeff()  # Ensure this function returns the correct coefficients
    # Unpack the parameters
    WF = parameters[0]
    HHb = parameters[1]
    HbO2 = parameters[2]
    a = parameters[3]
    b = parameters[4]
    
    rho = 30  # Source-detector separation
    
    # Convert xdata to corresponding indices for extinction coefficients
    indices = (xdata - 650).astype(int)  # Adjust indexing

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, coeff.shape[0] - 1)
    
    # Calculate the absorption coefficient (mua)
    mua = np.log(10) * HbO2 * coeff[indices, 2] + np.log(10) * HHb * coeff[indices, 1] + WF * coeff[indices, 3]
    
    # Calculate the reduced scattering coefficient (mus)
    mus = a * xdata ** (-b)
    
    # Effective attenuation coefficient (mue)
    mue = np.sqrt(3 * mua * (mua + mus))
    
    # Diffusion coefficient (D)
    D = 1.0 / (3 * (mua + mus))
    
    # Extrapolated boundary conditions
    z0 = 1.0 / (mua + mus)
    zb = ((1 + 0.493) / (1 - 0.493)) * 2 * D  # 0.493 is Reff at n=1.4
    
    # Distance calculations
    x0 = z0 ** 2 + rho ** 2
    x1 = (z0 + 2 * zb) ** 2 + rho ** 2
    
    # Model reflectance calculation
    model_refl = (1 / (4 * np.pi)) * ((z0 * (mue + 1.0 / np.sqrt(x0)) * np.exp(-mue * np.sqrt(x0)) / x0) +
                                      ((z0 + 2 * zb) * (mue + 1.0 / np.sqrt(x1)) * np.exp(-mue * np.sqrt(x1)) / x1))
    
    return model_refl  # Return the model reflectance without derivatives
