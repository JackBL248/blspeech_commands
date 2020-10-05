import argparse

parser = argparse.ArgumentParser(
    description="Train different CNN architectures on Google Speech Commands")
# ========================= Data Configs ==========================
parser.add_argument('--train-folder', default="toy_dataset/train", type=str,
                    help='path to train folder')
parser.add_argument('--val-folder', default="toy_dataset/val", type=str,
                    help='path to val folder')
parser.add_argument('--test-folder', default="toy_dataset/test", type=str,
                    help='path to test folder')

# ========================= Model Configs ==========================
parser.add_argument('--model-arch', default="resnet18", type=str,
                    help='Type of CNN architecture')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

# ========================= Log Configs ==========================
parser.add_argument('--log-file', default="log.txt", type=str,
                    help='Destination file for logging results')
