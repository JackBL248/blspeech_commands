import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path

from args import parser
from dataset import spectrogramDataset
from helper import get_normalise_coefficients
from model import Model
from preproc import (
    data_from_folder,
    normaliseSpectrogram,
    resizeSpectrogram,
    flipDimensions
)


def main():
    args = parser.parse_args()

    trainfolder = Path.joinpath(args.datafolder, "train")
    valfolder = Path.joinpath(args.datafolder, "val")
    testfolder = Path.joinpath(args.datafolder, "test")

    # extract train, val and test data
    train_preds, train_labels = data_from_folder(trainfolder, args.delta)
    val_preds, val_labels = data_from_folder(valfolder, args.delta)
    test_preds, test_labels = data_from_folder(testfolder, args.delta)

    if args.verbose:
        print("data extracted\n")

    # get means and std from training predictors
    train_means, train_stds = get_normalise_coefficients(train_preds)

    if args.verbose:
        print("training data means:")
        print(train_means)
        print("training data stds:")
        print(train_stds)

    # define transforms
    transform = transforms.Compose([
        normaliseSpectrogram(train_means, train_stds),
        flipDimensions(),
        resizeSpectrogram(),
        transforms.ToTensor(),
    ])
    # create dictionary of dataloaders and datasizes for train, val and test
    train_dataset = spectrogramDataset(train_preds, train_labels, transform)
    val_dataset = spectrogramDataset(val_preds, val_labels, transform)
    test_dataset = spectrogramDataset(test_preds, test_labels, transform)

    train_dataloader = data.DataLoader(
        train_dataset, args.batch,
        True, num_workers=args.workers
    )
    val_dataloader = data.DataLoader(
        val_dataset, args.batch,
        False, num_workers=args.workers
    )
    test_dataloader = data.DataLoader(
        test_dataset, args.batch,
        False, num_workers=args.workers
    )

    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
    }

    if args.verbose:
        print("dataloaders created\n")

    # set CUDA as device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model architecture, dropout, cost function and optimizer
    model = Model(args.model, args.dropout, device, args.log)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr)

    # write the model arch, lr and dropout to file
    with open(args.log, "a+") as f:
        f.write("model architecture:%s learning rate:%.3f dropout:%.2f\n" % (
            args.model,
            args.lr,
            args.dropout
            )
        )
    model.train(
        criterion,
        optimizer,
        dataloaders,
        args.epochs,
        args.patience
    )

    model.test(test_dataloader)


if __name__ == '__main__':
    main()
