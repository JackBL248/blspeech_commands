import numpy as np
from skimage.transform import resize

from config import Config


def resize_preds(preds_list, x=224, y=224):
    '''
    Resizes all predictors to 224,224 for input to model
    -----------------------
    Input: List of predictors, new x-dimension size, new y-dimension size

    Output: List of resized predictors
    -----------------------
    '''
    new_preds_list = []
    for preds in preds_list:
        new_preds = resize(preds, (x, y))
        new_preds_list.append(new_preds)

    return np.array(new_preds_list)


def get_class_from_filename(filename):
    '''
    Extracts class label from filename
    -----------------------
    Input: filename

    Output: Class label
    -----------------------
    '''
    # take the str class name from the filename
    str_label = filename.split("_")[0]
    # get the integer label
    label = Config.CLASSES_DICT[str_label]
    return label
