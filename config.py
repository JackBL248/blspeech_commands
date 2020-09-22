class Config():

    # Data vars
    with open("classes.txt", "r") as f:
        classes = [command.strip() for command in f.readlines()]
    CLASSES_DICT = {command[1]: command[0] for command in enumerate(classes)}
    NUM_CLASSES = len(CLASSES_DICT)

    LONGEST_SAMPLE = 16000

    # Spectrogram vars
    WINDOW_SIZE = 1024
    FFT_SIZE = 1024
    HOP_SIZE = 512
    WINDOW_TYPE = 'hann'
    FS = 16000
