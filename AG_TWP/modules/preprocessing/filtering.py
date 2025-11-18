import numpy as np
from scipy import signal

def get_notch_filter_coef(fs):
    # Notch filter at 50 Hz
    w0 = [49.0/(fs/2), 51.0/(fs/2)]
    return signal.butter(2, w0, btype='bandstop') # orden 8 -> N=4

def get_bandpass_filter_coef(fs):
    from math import pi
    Fpa = 1     # High pass cut-off frequency in Hz
    Fpb = 40    # Low pass cut-off frequency in Hz
    C1 = 100e-9 
    C2 = 1e-9
    R1 = 1 / (2 * pi * C1 * Fpa) 
    R2 = 1 / (2 * pi * C2 * Fpb)
    a = [1, (1 / (C1*R2)) + (1 / (C1*R1)) + (1 / (C2*R2)), (1 / (C1*C2*R1*R2))] # Denominator H(s)
    b = [1 / (C2*R2), 0] # Numerator H(s)
    return signal.bilinear(b, a, fs)

def get_bandpass_coef(N, low, high, fs):
    return signal.butter(N, [low, high], btype='band', fs=fs)

def preprocess_signal(senial, fs, filtfilt=False, band=[1, 40]):   
    b_notch, a_notch = get_notch_filter_coef(fs)
    b_bandpass, a_bandpass = get_bandpass_coef(N=1, low=band[0], high=band[1], fs=250)
    if filtfilt:
        senial = signal.filtfilt(b_notch, a_notch, senial)
        senial = signal.filtfilt(b_bandpass, a_bandpass, senial)
    else:
        senial = signal.lfilter(b_notch, a_notch, senial) # 50 hz
        senial = signal.lfilter(b_bandpass, a_bandpass, senial) # bandpass
    return senial

def preprocess_signal_im_tention(senial, fs, filtfilt=False, band=[1, 40]):   
    # IM-tention needs a Notch filter at 50 Hz
    b_notch, a_notch = get_notch_filter_coef(fs)
    
    if band != [1, 40]:
        b_bandpass, a_bandpass = get_bandpass_coef(N=4, low=band[0], high=band[1], fs=fs)
    else:
        b_bandpass, a_bandpass = get_bandpass_filter_coef(fs)
    if filtfilt:
        senial = signal.filtfilt(b_notch, a_notch, senial)
        senial = signal.filtfilt(b_bandpass, a_bandpass, senial)
    else:
        senial = signal.lfilter(b_notch, a_notch, senial) # 50 hz
        senial = signal.lfilter(b_bandpass, a_bandpass, senial) # bandpass
    return senial