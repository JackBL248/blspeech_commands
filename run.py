import torch
import torch.nn as nn
import torch.optim as optim

from preproc import prepare_dataset, data_from_folder
from args import parser
from model import Model


def main():
    args = parser.parse_args()

    # extract train, val and test data
    train_preds, train_labels = data_from_folder(args.train_folder)
    val_preds, val_labels = data_from_folder(args.val_folder)
    test_preds, test_labels = data_from_folder(args.test_folder)

    if args.verbosity:
        print("data extracted\n")

    # create dictionary of dataloaders and datasizes for train, val and test
    train_dataloader, train_datasize = prepare_dataset(
        train_preds,
        train_labels,
        args.batch_size,
        True
    )
    val_dataloader, val_datasize = prepare_dataset(
        val_preds,
        val_labels,
        args.batch_size,
        False
    )
    test_dataloader, test_datasize = prepare_dataset(
        test_preds,
        test_labels,
        args.batch_size,
        False
    )
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
    }
    datasizes = {
        "train": train_datasize,
        "val": val_datasize,
    }

    if args.verbosity:
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
