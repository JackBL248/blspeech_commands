import numpy as np
from skimage.transform import resize


def resize_preds(preds_list, x=224, y=224):
    '''
    Function resizing all predictors to 224,224 for input to resnet
    -----------------------
    Input: List of predictors, new x-dimension size, new y-dimension size

    Output: List of resized predictors
    -----------------------
    '''
    new_preds_list = []
    for preds in preds_list:
        new_preds = resize(preds, (224, 224))
        new_preds_list.append(new_preds)

    return np.array(new_preds_list)
