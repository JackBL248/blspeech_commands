import numpy as np
import os
import librosa
import torch
from torch import utils

from spectrogram import delta_spec, extract_features, preemphasis
from helper import resize_preds


# vars
LONGEST_SAMPLE = 16000

with open("classes.txt", "r") as f:
    CLASSES = [command.strip() for command in f.readlines()]
CLASSES_DICT = {command[1]: command[0] for command in enumerate(CLASSES)}


def preprocess_datapoint(
    wavfile,
    classes_dict=CLASSES_DICT,
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
        feats
    command = wavfile.split("_")[0]
    label = classes_dict[command]
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


def prepare_dataset(
    train_preds, train_labels,
    val_preds, val_labels,
    test_preds, test_labels
):
    # Resize data
    train_preds = resize_preds(train_preds)
    val_preds = resize_preds(val_preds)
    test_preds = resize_preds(test_preds)
    # Convert data to tensor
    train_preds = torch.Tensor(train_preds)
    train_labels = torch.Tensor(train_labels)
    val_preds = torch.Tensor(val_preds)
    val_labels = torch.Tensor(val_labels)
    test_preds = torch.Tensor(test_preds)
    test_labels = torch.Tensor(test_labels)
    # Convert labels to type long
    train_labels = torch.Tensor.long(train_labels)
    val_labels = torch.Tensor.long(val_labels)
    test_labels = torch.Tensor.long(test_labels)
    # Create training, validation and test datasets
    train_dataset = utils.TensorDataset(train_preds, train_labels)
    val_dataset = utils.TensorDataset(val_preds, val_labels)
    test_dataset = utils.TensorDataset(test_preds, test_labels)
    # Create training, validation and test datasets
    train_dataloader = utils.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True)
    val_dataloader = utils.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False)
    test_dataloader = utils.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False)
    # Create dictionary with train_dataloader and val_dataloader
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }
    # Create dictionary with dataset sizes
    dataset_sizes = {
        "train": len(train_preds),
        "val": len(val_preds),
        "test": len(test_preds)
    }
    return dataloaders, dataset_sizes
