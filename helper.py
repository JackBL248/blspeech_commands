import numpy as np

from config import Config


def get_class_from_filename(filename):
    '''
    Extracts class label from filename
    -----------------------
    Input: filename

    Output: Class label
    -----------------------
    '''
    # make sure filename is string
    if not isinstance(filename, str):
        filename = str(filename)
    # make sure to get only the filename and no the name of any parent folder
    if len(filename.split("/")) > 1:
        filename = filename.split("/")[-1]
    # take the str class name from the filename
    str_label = filename.split("_")[0]
    # get the integer label
    label = Config.CLASSES_DICT[str_label]
    return label


def get_normalise_coefficients(train_preds):
    '''
    Finds the mean and std for channels from th
    -----------------------
    Args: training predictors (list)

    Returns: mean and std for each channel
    -----------------------
    '''
    training_mean = np.mean(train_preds, axis=(0, 2, 3))
    training_std = np.std(train_preds, axis=(0, 2, 3))
    return training_mean, training_std
