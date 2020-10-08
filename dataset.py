import torch.utils.data as data


class spectrogramDataset(data.Dataset):
    def __init__(self, preds, labels, transform):
        self.preds = preds
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # prepare spectrogram for input into model
        spectrogram = self.transform(self.preds[index])
        # convert label to torch.Tensor
        return spectrogram, float(self.labels[index])

    def __len__(self):
        return len(self.labels)
