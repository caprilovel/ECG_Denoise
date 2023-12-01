import pywt 
import numpy as np

import pandas as pds 


def wavelet_denoise(ecg_data):
    """using wavelet to denoise the ecg data

    Args:
        ecg_data (numpy.ndarray): the input ecg data, should be a 2d nparray or 3d nparray. 

    Returns:
        np.array: the denoised ecg data, have the same shape as the input ecg data.
    """
    import pywt
    w = pywt.Wavelet('db8')
    if len(ecg_data.shape) == 2:
        datarec = []
        for data in ecg_data:
            
            maxlev = pywt.dwt_max_level(len(data), w.dec_len)
            threshold = 0.04
            coeffs = pywt.wavedec(data, 'db8', level=maxlev)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            datarec.append(pywt.waverec(coeffs, 'db8'))
        return np.array(datarec)
    elif len(ecg_data.shape) == 3:
        datarec = []
        for data in ecg_data:
            datarec.append(wavelet_denoise(data))
        return np.array(datarec)


def fft_denoise(ecg_datas, threshold=0.04):
    '''denoise via frequency fourier transform
    
      params:
        ecg_datas: the input ecg data, should be a 2D nparray or a list of 1D nparrays.
        alpha: meta-param for denoise, the threshold of the noise frequency which is the mean frequency. default to be 1.
        
      Output:
        denoised_data: a 2D ndarray which has the same shape as the input ecg data. 
    '''
    if isinstance(ecg_datas, list):
        ecg_datas = np.array(ecg_datas)
        
    new_ecg_data = []
    for ecg_data in ecg_datas:
        # Apply FFT to the input data
        ecg_fft = fft(ecg_data)

        # Calculate the magnitude of the FFT coefficients
        magnitude = np.abs(ecg_fft)

        # Find the threshold for noise reduction
        cutoff = threshold * np.max(magnitude)

        # Set coefficients below the threshold to zero (remove noise)
        ecg_fft[magnitude < cutoff] = 0

        # Reconstruct the signal using inverse FFT
        denoised_ecg = ifft(ecg_fft)
        new_ecg_data.append(denoised_ecg.real)
    return np.array(new_ecg_data)