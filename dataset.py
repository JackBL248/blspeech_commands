import torch
import torch.utils.data as data


class spectrogramDataset(data.Dataset):
    def __init__(self, preds, labels, transform):
        self.preds = preds
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # prepare spectrogram for input into model
        print(self.labels[index])
        spectrogram = self.transform(self.preds[index])
        # convert label to torch.Tensor
        label = torch.Tensor([self.labels[index]])
        label = torch.Tensor.long(label)
        print(label)
        return spectrogram, label

    def __len__(self):
        return len(self.labels)
