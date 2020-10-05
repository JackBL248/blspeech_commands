import argparse

parser = argparse.ArgumentParser(description="Train different CNN architectures on Google Speech Commands")
# ========================= ModelConfigs ==========================
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
