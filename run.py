import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms

from args import parser
from dataset import spectrogramDataset
from helper import get_normalise_coefficients
from model import Model
from preproc import data_from_folder


def main():
    args = parser.parse_args()

    # extract train, val and test data
    train_preds, train_labels = data_from_folder(args.train_folder, args.delta)
    val_preds, val_labels = data_from_folder(args.val_folder, args.delta)
    test_preds, test_labels = data_from_folder(args.test_folder, args.delta)

    if args.verbose:
        print("data extracted\n")

    # get means and std from resized training predictors
    train_means, train_stds = get_normalise_coefficients(train_preds)

    if args.verbose:
        print("training data means:")
        print(train_means)
        print("training data stds:")
        print(train_stds)

    # define transforms
    transform = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize(mean=train_means, std=train_stds),
        transforms.Resize(224),
    )
    # create dictionary of dataloaders and datasizes for train, val and test
    train_dataset = spectrogramDataset(train_preds, train_labels, transform)
    val_dataset = spectrogramDataset(val_preds, val_labels, transform)
    test_dataset = spectrogramDataset(test_preds, test_labels, transform)

    train_dataloader = data.DataLoader(train_dataset, args.batch_size, True)
    val_dataloader = data.DataLoader(val_dataset, args.batch_size, False)
    test_dataloader = data.DataLoader(test_dataset, args.batch_size, False)

    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
    }
    datasizes = {
        "train": len(train_dataloader),
        "val": len(val_dataloader),
    }

    if args.verbose:
        print("dataloaders created\n")

    # set CUDA as device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model architecture, dropout, cost function and optimizer
    model = Model(args.model_arch, args.dropout, device, args.log)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.params(), lr=args.lr)

    # write the model arch, lr and dropout to file
    with open(args.log, "a+") as f:
        f.write("model_arch:%s learning_rate:%.3f dropout:%.2f\n" % (
            args.model_arch,
            args.lr,
            args.dropout
            )
        )
    model.train(
        criterion,
        optimizer,
        dataloaders,
        datasizes,
        args.num_epochs,
        args.patience
    )

    model.test(test_dataloader)


if __name__ == '__main__':
    main()
