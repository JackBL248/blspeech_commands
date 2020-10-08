import numpy as np
import librosa
from pathlib import Path

from config import Config
from spectrogram import delta_spec, extract_features, preemphasis
from helper import get_class_from_filename


def preprocess_datapoint(
    wavfile,
    classes_dict=Config.CLASSES_DICT,
    deltas=False
):
    '''
    Standardise the length of the audio,
    Apply preemphasis (two filters),
    Generate predictors (stft) and targets
    and resize predictors
    -----------------------
    Args: wavfile (path)

    Returns: features (np.array), label (int)
    -----------------------
    '''
    # Read in audio
    raw_samples, Fs = librosa.core.load(wavfile, sr=None)
    # Zero pad audio to the longest audio file
    samples = np.zeros(Config.LONGEST_SAMPLE)
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
        feats
    label = get_class_from_filename(wavfile)
    return feats, label


def data_from_folder(folder_path, deltas=False):
    '''
    Get preprocessed data and labels from folder.
    -----------------------
    Args: folder path (str)

    Returns: predictors and labels
    -----------------------
    '''
    data_path = Path(folder_path)
    audiofiles = data_path.glob("*wav")
    # Initialise predictor and label lists
    preds_list = []
    labels_list = []
    # iterate through files and append preds and labels to lists
    for audiof in audiofiles:
        preds, label = preprocess_datapoint(audiof, deltas=deltas)
        preds_list.append(preds)
        labels_list.append(label)
    return preds_list, labels_list


class normaliseSpectrogram(object):
    '''
    Functor for normalising each channel of the spectrogram.
    -----------------------
    Attrs:
    - means (np.array, should be shape (3, 1, 1))
    - stds (np.array, should be shape (3, 1, 1))
    -----------------------
    '''
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, spec):
        spec = (spec - self.means) / self.stds
        return spec
