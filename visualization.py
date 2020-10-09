import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate, resize

from config import Config
from preproc import preprocess_datapoint


def view_spec(spec, label):
    '''
    Views spectrogram using matplotlib.
    -----------------------
    Input: Spectrogram (np.array)
    -----------------------
    '''
    spec = spec.transpose(2, 1, 0)
    spec = resize(spec, (224, 224))
    spec -= np.min(spec)
    spec /= np.max(spec)

    plt.imshow(spec)
    plt.title(label)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def view_spec_from_file(wavfile):
    '''
    Extracts and views the spectrogram from an audio file.
    -----------------------
    Input: wavfile (str)
    -----------------------
    '''
    # Process data
    print(wavfile)
    spec, label = preprocess_datapoint(wavfile)
    str_label = Config.CLASSES_DICT_INVERTED[label]

    # # view spectrogram
    view_spec(spec, str_label)
