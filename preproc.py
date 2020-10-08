import numpy as np
import librosa
import torch
from pathlib import Path
from torch import utils

from config import Config
from spectrogram import delta_spec, extract_features, preemphasis
from helper import get_class_from_filename, resize_preds


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
    Input: wavfile

    Output: features, integer label
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


def data_from_folder(folder_path):
    '''
    Get preprocessed data and labels from folder.
    -----------------------
    Input: folder path

    Output: predictors and labels
    -----------------------
    '''
    data_path = Path(folder_path)
    audiofiles = data_path.glob("*wav")
    # Initialise predictor and label lists
    preds_list = []
    labels_list = []
    # iterate through files and append preds and labels to lists
    for audiof in audiofiles:
        preds, label = preprocess_datapoint(audiof)
        preds_list.append(preds)
        labels_list.append(label)
    return preds_list, labels_list


def find_normalise_coefficients(train_preds):
    '''
    Finds the mean and std for channels from th
    -----------------------
    Input: training predictors

    Output: mean and std for each channel
    -----------------------
    '''
    training_mean = np.mean(train_preds, axis=(0, 2, 3))
    training_std = np.std(train_preds, axis=(0, 2, 3))
    return training_mean, training_std


def prepare_dataset(preds, labels, batch_size, shuffle=True):
    '''
    Create PyTorch data loaders from preds and labels
    -----------------------
    Args: predictors and labels, batch_size(int), shuffle(boolean)

    Output: dataloader
    -----------------------
    '''
    # Resize data
    preds = resize_preds(preds)
    # Convert data to tensor
    preds = torch.Tensor(preds)
    labels = torch.Tensor(labels)
    # Convert labels to type long
    labels = torch.Tensor.long(labels)
    # Create dataset and dataloader
    dataset = utils.TensorDataset(preds, labels)
    dataloader = utils.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader
