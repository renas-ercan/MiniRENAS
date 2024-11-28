import numpy as np
from scipy.signal import savgol_filter
from load_extinction_coeff import load_extinction_coeff

def gen_refl_model_diff2_HHb(parameters, xdata, WF):
    """
    Generates the second derivative of the model reflectance for HHb fitting.
    """
    coeff = load_extinction_coeff()
    rho = 30  # Source-detector separation

    # Unpack parameters
    # WF is fixed and passed separately
    HHb = parameters[1]
    HbO2 = parameters[2]
    a = parameters[3]
    b = parameters[4]

    # Indices to read off the table (adjust for Python indexing)
    indices = (xdata - 650)
    indices = indices.astype(int)

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, coeff.shape[0] - 1)

    # Calculate absorption coefficient (mua)
    mua = (np.log(10) * HbO2 * coeff[indices, 2] +
           np.log(10) * HHb * coeff[indices, 1] +
           WF * coeff[indices, 3])

    # Calculate reduced scattering coefficient (mus)
    mus = a * xdata ** (-b)

    # Effective attenuation coefficient (mue)
    mue = np.sqrt(3 * mua * (mua + mus))

    # Diffusion coefficient (D)
    D = 1.0 / (3 * (mua + mus))

    # Extrapolated boundary conditions
    z0 = 1.0 / (mua + mus)
    Reff = 0.493  # Reff at n=1.4
    zb = ((1 + Reff) / (1 - Reff)) * 2 * D

    # Distance calculations
    x0 = z0 ** 2 + rho ** 2
    x1 = (z0 + 2 * zb) ** 2 + rho ** 2

    # Model reflectance calculation
    model_refl = (1 / (4 * np.pi)) * (
        (z0 * (mue + 1.0 / np.sqrt(x0)) * np.exp(-mue * np.sqrt(x0)) / x0) +
        ((z0 + 2 * zb) * (mue + 1.0 / np.sqrt(x1)) * np.exp(-mue * np.sqrt(x1)) / x1)
    )

    # Smooth the model reflectance
    model_refl_smoothed = savgol_filter(model_refl, window_length=11, polyorder=3)

    # First and second derivatives
    diff_1_model_refl = np.diff(model_refl_smoothed)
    diff_1_model_refl_smoothed = savgol_filter(diff_1_model_refl, window_length=5, polyorder=2)
    diff_2_model_refl = np.diff(diff_1_model_refl_smoothed)

    return diff_2_model_refl
