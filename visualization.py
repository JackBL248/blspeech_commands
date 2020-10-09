import matplotlib.pyplot as plt
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
    str_label = Config.CLASSES_DICT_INVERTED[label]
    spec_copy = spec.transpose(2, 1, 0)
    spec_copy = resize(spec, (224, 224))
    spec_copy = rotate(spec_copy, 90)

    plt.imshow(spec_copy)
    plt.title(str_label)
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
    spec, label = preprocess_datapoint(wavfile)
    # view spectrogram
    view_spec(spec)
