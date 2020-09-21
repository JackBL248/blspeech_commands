import numpy as np
import os
import essentia
import essentia.standard as ess
import librosa
from scipy.signal import butter, filtfilt


# vars
WINDOW_SIZE = 1024
FFT_SIZE = 1024
HOP_SIZE = 512
WINDOW_TYPE = 'hann'
FS = 16000
LONGEST_SAMPLE = 16000

with open("classes.txt", "r") as f:
    CLASSES = [command.strip() for command in f.readlines()]
CLASSES_DICT = {command[1]: command[0] for command in enumerate(CLASSES)}


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
    M=WINDOW_SIZE,
    N=FFT_SIZE,
    H=HOP_SIZE,
    fs=FS,
    window_type=WINDOW_TYPE
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
    SP = SP[:, :int(FFT_SIZE/4+1)]

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


def preprocess_datapoint(
    wavfile, label, classes_dict=CLASSES_DICT, deltas=False
):
    '''
    Function for standardising the length of the audio,
    applying preemphasis (two filters),
    generating predictors (stft) and targets
    -----------------------
    Input: wavfile

    Output: Features, one-hot label
    -----------------------
    '''
    # Read in audio
    raw_samples, Fs = librosa.core.load(wavfile, sr=None)

    # Zero pad audio to the longest audio file
    samples = np.zeros(LONGEST_SAMPLE)
    samples[:len(raw_samples)] = raw_samples

    # Apply preemphasis
    e_samples = preemphasis(samples, Fs)
    # Extract features (spectrogram)
    spec = extract_features(e_samples, fs=Fs)
    # Create two extra channels...
    if deltas:  # using derivative and double derivative
        d_spec = delta_spec(spec)
        dd_spec = delta_spec(d_spec)
        feats = np.array([spec, d_spec, dd_spec])
    else:  # by using the same spectrogram for each of the three channels
        feats = np.array([spec, spec, spec])
    label = classes_dict[label]

    return feats, label


def split_data(data_path, files, commands):

    '''
    Function for splitting audio files into train, validation and test sets
    -----------------------
    Input: Folder path with all data

    Output: Train, validation and test predictors and labels
    -----------------------
    '''

    # Initialise predictor and label lists
    preds_list = []
    labels_list = []
    # iterate through files and commands lists
    for audio_file, command in zip(files, commands):
        audio_path = os.path.join(data_path, audio_file)
        # Preprocess audio file
        feats, labels = preprocess_datapoint(audio_path, command)
        # Append features and labels to relevant list
        preds_list.append(feats)
        labels_list.append(labels)

    return preds_list, labels_list


def normalise_datasets(train_preds, val_preds, test_preds):

    '''
    Function normalising all predictors based on the mean and standard
    deviation of the training predictors
    -----------------------
    Input: Train, validation and test predictors

    Output: Normalised train, validation and test predictors
    -----------------------
    '''
    # Calculate mean and standard deviation of training predictors
    tr_mean = np.mean(train_preds)
    tr_std = np.std(train_preds)

    # Subtract mean and divide by standard deviation for normalised predictors
    train_preds = np.subtract(train_preds, tr_mean)
    train_preds = np.divide(train_preds, tr_std)
    val_preds = np.subtract(val_preds, tr_mean)
    val_preds = np.divide(val_preds, tr_std)
    test_preds = np.subtract(test_preds, tr_mean)
    test_preds = np.divide(test_preds, tr_std)

    return train_preds, val_preds, test_preds
