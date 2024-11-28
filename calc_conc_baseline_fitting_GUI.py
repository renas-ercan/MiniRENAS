import numpy as np
from tissue_specific_extinction_coefficient_650to1000 import tissue_specific_extinction_coefficient_650to1000
from DPF_Lambda_Dependency_740to915 import DPF_Lambda_Dependency_740to915
from scipy.interpolate import PchipInterpolator

from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from generateexp_refl import generateexp_refl
from gen_refl_model_diff2_WF import gen_refl_model_diff2_WF
from gen_refl_model_diff2_HHb import gen_refl_model_diff2_HHb
from gen_refl_model_diff1_HbO2 import gen_refl_model_diff1_HbO2

calc_counter = 0

def calc_conc_baseline_fitting_GUI_Sim(ref_spectra, input_spectra, input_wavelength, integration_time, reference_file, type, age, dpf_value, optode_dist):
    
    global calc_counter
    calc_counter += 1 # Increment the counter each time the function is called

    # Define a small positive value to replace zeros
    replacement_value = np.finfo(float).eps
    epsilon = 1e-6

    # Replace zeros in input_spectra in the GUI code
    input_spectra[input_spectra == 0] = replacement_value

    # Ensure ref_spectra and input_spectra are NumPy arrays
    ref_spectra = np.array(ref_spectra).flatten()
    input_spectra = np.array(input_spectra).flatten()

    # Load data from the reference file
    reference_file = 'Ibsen_ref_Raw_No_ND_Filter_Int_Time_9ms.npy'
    ref1 = np.load(reference_file, allow_pickle=True)  # NumPy Array

    # Ensure both spectra have the same number of elements (wavelength points)
    if len(ref_spectra) != len(input_spectra):
        raise ValueError("ref_spectra and input_spectra must have the same number of wavelength points.")

    # Parameters
    WaterFraction = 0.85
    raw = input_spectra.T
    PL_method = 2
    wl_cal = input_wavelength


    # Age-dependent DPF
    user_age = age
    # DPF = 5.11 + 0.106 * (user_age ** 0.723)
    user_dpf = dpf_value

    print(f"Running concentration calculations with Age: {user_age}, DPF: {user_dpf}")

    # Wavelengths used to resolve over
    wl = np.arange(780, 901)

    # Extinction coefficients
    c = tissue_specific_extinction_coefficient_650to1000()
    ext_coeffs = c[wl[0] - 650 : wl[-1] - 650 + 1, [2, 3, 4]]
    ext_coeffs_inv = np.linalg.pinv(ext_coeffs)

    # DPF dependency
    DPF_dep = DPF_Lambda_Dependency_740to915()
    DPF_dep = DPF_dep[wl[0] - 740 : wl[-1] - 740 + 1, 1]

    input_wavelength = np.squeeze(input_wavelength)

    # Ensure input_spectra is 2-dimensional
    #input_spectra = np.atleast_2d(input_spectra)  

    #print(f'ref_spectra', ref_spectra)
    #print(f'input_spectra', input_spectra)

    # Broadband Fitting Algorithm
    Abs = {'WF': [], 'HHb': [], 'HbO2': [], 'a': [], 'b': []}

    y = input_spectra
    x = ref1[0, :]  # Access the first row of the array


    if PL_method == 2:
        # Wave start/end points for reflectance models
        wave_start, wave_end = 825, 850
        Boundaries = np.array([[0.95, 60, 60, 40, 1], [0.7, 10, 10, 1, 0.1]])

        # Generate experimental reflectance
        exp_refl, diff_1_exp_refl, diff_2_exp_refl = generateexp_refl(wl_cal, x, y, wave_start, wave_end)

        # Apply the Savitzky-Golay filter and calculate the second derivative (diff of diff)
        exp_refl_smoothed = savgol_filter(exp_refl, window_length=5, polyorder=2)
        first_diff = np.diff(exp_refl_smoothed)
        first_diff_smoothed = savgol_filter(first_diff, window_length=5, polyorder=2)
        diff_2_exp_refl = np.diff(first_diff_smoothed)

        # Generate the xdata range
        xdata = np.arange(wave_start, wave_end + 1)

        # Ensure lengths match
        if len(exp_refl_smoothed) != len(xdata):
            print("Adjusting exp_refl_smoothed to match xdata length.")
            exp_refl_smoothed = exp_refl_smoothed[:len(xdata)]

        # Extract boundary values
        lb = Boundaries[1, :]
        ub = Boundaries[0, :]

        def diff_2_model_refl_WF(parameters, xdata):
            model_refl = gen_refl_model_diff2_WF(parameters, xdata)
            model_refl_smoothed = savgol_filter(model_refl, window_length=5, polyorder=2)
            first_diff_model = np.diff(model_refl_smoothed)
            first_diff_model_smoothed = savgol_filter(first_diff_model, window_length=5, polyorder=2)
            diff_2_model_refl = np.diff(first_diff_model_smoothed)
            return diff_2_model_refl

        # Define initial parameters and bounds
        initial_params = [0.75, 10, 20, 30, 0.6]

        # Define optimization options
        options = {
            'ftol': 1e-10,   # Function tolerance
            'xtol': 1e-10,   # Step tolerance
            'max_nfev': 100000,  # Maximum function evaluations
        }

        # Test the model function to ensure lengths match
        test_model_diff2 = diff_2_model_refl_WF(initial_params, xdata)

        # Ensure the lengths match before optimization
        min_length = min(len(diff_2_exp_refl), len(test_model_diff2))
        diff_2_exp_refl = diff_2_exp_refl[:min_length]
        test_model_diff2 = test_model_diff2[:min_length]

        # Perform the curve fitting using least_squares
        result = least_squares(
            fun=lambda params: diff_2_model_refl_WF(params, xdata)[:min_length] - diff_2_exp_refl,
            x0=initial_params,
            bounds=(lb, ub),
            **options
        )

        # Extract the fitting parameters
        fit_parameters_1 = result.x
        WF = fit_parameters_1[0]

        # Append results to Abs dictionary
        Abs['WF'].append(WF)

        # Second Fitting
        wave_start = 700
        wave_end = 800

        exp_refl, diff_1_exp_refl, diff_2_exp_refl = generateexp_refl(wl_cal, x, y, wave_start, wave_end)

        # Apply the Savitzky-Golay filter and calculate the second derivative (diff of diff)
        exp_refl_smoothed = savgol_filter(exp_refl, window_length=5, polyorder=2)
        first_diff = np.diff(exp_refl_smoothed)
        first_diff_smoothed = savgol_filter(first_diff, window_length=5, polyorder=2)
        diff_2_exp_refl = np.diff(first_diff_smoothed)

        # Generate the xdata range
        xdata = np.arange(wave_start, wave_end + 1)

        # Ensure lengths match
        if len(exp_refl_smoothed) != len(xdata):
            exp_refl_smoothed = exp_refl_smoothed[:len(xdata)]

        # Adjust bounds to fix WF
        lb[0] = WF - epsilon
        ub[0] = WF + epsilon

        # Ensure initial_params[0] is WF
        initial_params = fit_parameters_1.copy()
        initial_params[0] = WF

        def diff_2_model_refl_HHb(parameters, xdata):
            model_refl = gen_refl_model_diff2_HHb(parameters, xdata, WF)
            model_refl_smoothed = savgol_filter(model_refl, window_length=5, polyorder=2)
            first_diff_model = np.diff(model_refl_smoothed)
            first_diff_model_smoothed = savgol_filter(first_diff_model, window_length=5, polyorder=2)
            diff_2_model_refl = np.diff(first_diff_model_smoothed)
            return diff_2_model_refl

        # Define optimization options
        options = {
            'ftol': 1e-10,   # Function tolerance
            'xtol': 1e-10,   # Step tolerance
            'max_nfev': 100000,  # Maximum function evaluations
        }

        # Test the model function to ensure lengths match
        test_model_diff2 = diff_2_model_refl_HHb(initial_params, xdata)

        # Ensure the lengths match before optimization
        min_length = min(len(diff_2_exp_refl), len(test_model_diff2))
        diff_2_exp_refl = diff_2_exp_refl[:min_length]
        test_model_diff2 = test_model_diff2[:min_length]

        # Perform the curve fitting using least_squares
        result = least_squares(
            fun=lambda params: diff_2_model_refl_HHb(params, xdata)[:min_length] - diff_2_exp_refl,
            x0=initial_params,
            bounds=(lb, ub),
            **options
        )

        # Extract the fitting parameters
        fit_parameters_2 = result.x
        HHb = fit_parameters_2[1]

        # Append results to Abs dictionary
        Abs['HHb'].append(HHb)

        # Third Fitting
        wave_start = 680
        wave_end = 850

        exp_refl, diff_1_exp_refl, diff_2_exp_refl = generateexp_refl(wl_cal, x, y, wave_start, wave_end)

        # Apply the Savitzky-Golay filter and calculate the first derivative
        exp_refl_smoothed = savgol_filter(exp_refl, window_length=5, polyorder=2)
        diff_1_exp_refl = np.diff(exp_refl_smoothed)

        # Generate the xdata range
        xdata = np.arange(wave_start, wave_end + 1)

        # Ensure lengths match
        if len(exp_refl_smoothed) != len(xdata):
            exp_refl_smoothed = exp_refl_smoothed[:len(xdata)]

        # Adjust bounds to fix HHb
        lb[1] = HHb - epsilon
        ub[1] = HHb + epsilon

        # Ensure initial_params[1] = HHb
        initial_params = fit_parameters_2.copy()
        initial_params[1] = HHb

        def diff_1_model_refl_HbO2(parameters, xdata):
            model_refl = gen_refl_model_diff1_HbO2(parameters, xdata, WF, HHb)
            model_refl_smoothed = savgol_filter(model_refl, window_length=5, polyorder=2)
            first_diff_model = np.diff(model_refl_smoothed)
            return first_diff_model

        # Define optimization options
        options = {
            'ftol': 1e-10,   # Function tolerance
            'xtol': 1e-10,   # Step tolerance
            'max_nfev': 100000,  # Maximum function evaluations
        }

        # Test the model function to ensure lengths match
        test_model_diff1 = diff_1_model_refl_HbO2(initial_params, xdata)

        # Ensure the lengths match before optimization
        min_length = min(len(diff_1_exp_refl), len(test_model_diff1))
        diff_1_exp_refl = diff_1_exp_refl[:min_length]
        test_model_diff1 = test_model_diff1[:min_length]

        # Perform the curve fitting using least_squares
        result = least_squares(
            fun=lambda params: diff_1_model_refl_HbO2(params, xdata)[:min_length] - diff_1_exp_refl,
            x0=initial_params,
            bounds=(lb, ub),
            **options
        )

        # Extract the fitting parameters
        fit_parameters_3 = result.x
        HbO2 = fit_parameters_3[2]
        a = fit_parameters_3[3]
        b = fit_parameters_3[4]

        # Append results to Abs dictionary
        Abs['HbO2'].append(HbO2)
        Abs['a'].append(a)
        Abs['b'].append(b)


    if PL_method == 2:
        DPF_water = (np.array(Abs['WF']) / WaterFraction) * 10 / optode_dist
        PL_water = DPF_water * optode_dist
    else:
        PL_water = []

    PL = PL_water

    # END OF BROADBAND FITTING

    # Calculate attenuation
    # Calculate attenuation more carefully and add debugging output
    attenuation = np.log10(ref_spectra / input_spectra)
    #print(f"Attenuation calculated at current time: {attenuation}")
    

    # Replace any -inf, inf, or NaN values
    # Ensure no extreme values interfere
    attenuation[np.isinf(attenuation)] = replacement_value
    attenuation[np.isnan(attenuation)] = replacement_value

    # Separate real and imaginary parts
    attenuation_real = np.real(attenuation)
    attenuation_imag = np.imag(attenuation)


    # Ensure input_wavelength is strictly increasing
    sorted_indices = np.argsort(input_wavelength)
    sorted_wavelength = input_wavelength[sorted_indices]
    sorted_attenuation_real = attenuation_real[sorted_indices]
    sorted_attenuation_imag = attenuation_imag[sorted_indices]


    # Perform PCHIP interpolation (equivalent to MATLAB's 'spline')
    spline_interp_real = PchipInterpolator(sorted_wavelength, sorted_attenuation_real)
    spline_interp_imag = PchipInterpolator(sorted_wavelength, sorted_attenuation_imag)

    # Interpolate to the desired wavelength range
    # Recalculate the baseline correctly before interpolation
    attenuation_interp_real = spline_interp_real(wl)
    attenuation_interp_imag = spline_interp_imag(wl)

    attenuation_interp = attenuation_interp_real + 1j * attenuation_interp_imag


    # Attenuation with wavelength dependency
    atten_int_wldep = attenuation_interp_real / DPF_dep

    # Calculate concentrations
    Conc = []
    for jj in range(ext_coeffs_inv.shape[0]):
        conc_value = (ext_coeffs_inv[jj, :] @ atten_int_wldep) * (1 / (optode_dist * user_dpf))
        Conc.append(conc_value)

    Conc = np.array(Conc)

    # Convert to concentrations in µM
    HbO2 = Conc[0] * 1000 * 1000
    HHb = Conc[1] * 1000 * 1000
    CCO = Conc[2] * 1000 * 1000

    # Calculate total hemoglobin (HbT)
    HbT = HbO2 + HHb

    # Calculate the elapsed time assuming measurements taken at integration_time intervals
    Time = [integration_time * calc_counter]

    return HbO2, HHb, CCO, Time, HbT
