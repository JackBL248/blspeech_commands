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
    Function for standardising the length of the audio,
    applying preemphasis (two filters),
    generating predictors (stft) and targets
    and resizing predictors
    -----------------------
    Input: wavfile

    Output: Features, one-hot label
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
    Input: Folder path

    Output: Predictors and labels
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
    tr_mean = np.mean(train_preds, axis=0)
    tr_std = np.std(train_preds, axis=0)
    # Subtract mean and divide by standard deviation for normalised predictors
    train_preds = np.subtract(train_preds, tr_mean)
    train_preds = np.divide(train_preds, tr_std)
    val_preds = np.subtract(val_preds, tr_mean)
    val_preds = np.divide(val_preds, tr_std)
    test_preds = np.subtract(test_preds, tr_mean)
    test_preds = np.divide(test_preds, tr_std)
    return train_preds, val_preds, test_preds


def prepare_dataset(preds, labels, batch_size, shuffle=True):
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
    # Get dataset size
    dataset_size = len(preds)
    return dataloader, dataset_size
