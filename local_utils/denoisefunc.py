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


def fft_denoise(ecg_datas, alpha=1):
    '''denoise via frequency fourier transform
    
      params:
        ecg_data: the input ecg data, should be a 2d nparray.
        alpha: meta-param for denoise, the threshold of the noise frequency which is the mean frequency. default to be 1.
        
      Output:
        denoised_data: a 2d ndarray which have the same shape as the input ecg data. 
    '''
    import numpy.fft as nf
    import numpy as np
    new_ecg_data = []
    for ecg_data in ecg_datas:
        fft_data = nf.fft(ecg_data)
        abs_fft_data = np.abs(fft_data)
        
        meanf = np.mean(abs_fft_data)
        noise_index = np.where(abs_fft_data <= alpha * meanf)[0]
        
        fft_data[noise_index] = 0
        new_ecg_data.append(nf.ifft(fft_data))
    return np.array(new_ecg_data)