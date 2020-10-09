# Speech Command Classification
Classification of speech commands from the Google Speech Commands dataset using spectrogram extraction for preprocessing, and offering the use of Alexnet [1], Resnet18, -32 and -50 [2], and Densenet121 [3].


## Data preparation
You can download the Google Speech Commands dataset from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data or from https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html. Alternatively, you can use the toy_dataset included in this repo.

The data should be split into train, validation and test folders, with the rootfolder in the main directory, e.g.:

|--blspeech_commands
|   |--data
|   |   |--train
|   |   |--val
|   |   |--test
|   |--README.md

etc.




## References

[1] Krizhevsky, A., Sutskever, I. and Hinton, G.E., 2012. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).

[2] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[3] Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K.Q., 2017. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).

