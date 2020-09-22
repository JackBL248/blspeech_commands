import numpy as np
import essentia
import essentia.standard as ess
from scipy.signal import butter, filtfilt

from config import Config


def preemphasis(input_vector, fs):
    '''
    Applies 2 simple high pass FIR filters in cascade to emphasize
    high frequencies and cut unwanted low-frequencies
    -----------------------
    Input: audio samples (1d numpy array), sampling frequency

    Output: Filtered audio
    ---------
    '''
    # first gentle high pass
    alpha = 0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero, past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present, past)

    # second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.)  # Normalize the frequency
    b, a = butter(6, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output


def extract_features(
    x,
    M=Config.WINDOW_SIZE,
    N=Config.FFT_SIZE,
    H=Config.HOP_SIZE,
    fs=Config.FS,
    window_type=Config.WINDOW_TYPE
):
    '''
    Function that extracts spectrogram from an audio signal
    -----------------------
    Input: Samples, window size (int), FFT size (int), Hop size (int),
    Sampling rate, Window type (e.g. Hanning)

    Output: Spectrogram
    -----------------------
    '''
    # init functions and vectors
    x = essentia.array(x)
    spectrum = ess.Spectrum(size=N)
    window = ess.Windowing(size=M, type=window_type)
    SP = []
    # compute STFT
    for frame in ess.FrameGenerator(
        x,
        frameSize=M,
        hopSize=H,
        startFromZero=True
    ):  # generate frames
        wX = window(frame)  # window frame
        mX = spectrum(wX)  # compute fft

        SP.append(mX)
    SP = essentia.array(SP)
    SP = np.power(SP, 2./3.)  # power law compression
    SP = SP[:, :int(Config.FFT_SIZE/4+1)]

    return SP


def delta_spec(spec, n=2):
    '''
    Calculates the delta spectrogram (change in frequency over time)
    from a spectrogram
    -----------------------
    Input: Spectrogram

    Output: Delta Spectrogram
    -----------------------
    '''
    # Pad spectrogram with zeros along time axis (beginning and end)
    spec_pad = np.pad(spec, ((n, n), (0, 0)), mode='constant')
    delta_spec = np.empty_like(spec)
    for frame in range(n, int(spec.shape[0])):
        delta_frame = np.sum(np.array([i*(spec_pad[frame+i] - spec_pad[frame-i]) for i in range(1, n+1)]), axis=0)
        delta_frame = delta_frame / (2*np.sum(np.array([i**2 for i in range(1, n+1)])))
        delta_spec[frame] = delta_frame

    return delta_spec
