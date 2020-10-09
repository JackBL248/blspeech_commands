import argparse

parser = argparse.ArgumentParser(
    description="Train different CNN architectures on Google Speech Commands")
# ========================= Data Configs ==========================
parser.add_argument('--trainfolder', default="toy_dataset/train", type=str,
                    help='path to train folder')
parser.add_argument('--valfolder', default="toy_dataset/val", type=str,
                    help='path to val folder')
parser.add_argument('--testfolder', default="toy_dataset/test", type=str,
                    help='path to test folder')

# ========================= Preprocess Configs ==========================
parser.add_argument('-d', '--delta', default=False, type=bool,
                    help='whether to use delta and double delta spectrograms\
                    to fill the two other channels')

# ========================= Model Configs ==========================
parser.add_argument('--model', default="resnet18", type=str,
                    help='Type of CNN architecture')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--patience', default=15, type=int,
                    help='early stopping patience')
parser.add_argument('-b', '--batch', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='model dropout')
parser.add_argument('--workers', default=4, type=int,
                    help='number of workers')

# ========================= Log Configs ==========================
parser.add_argument('--log', default="log.txt", type=str,
                    help='Destination file for logging results')
parser.add_argument('-v', '--verbose', default=False, type=bool,
                    help='whether to print updates on data preprocessing')
