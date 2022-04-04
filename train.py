import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import torch.backends.cudnn as cudnn
import torch.optim
import train_baseline
import train_regression
import train_softargmax
import train_skeleton

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def set_method(parser):
    parser.add_argument(
        '--heatmap',
        default=False,
        action='store_true',
        help='true if using the baseline heatmap model'
    )

    parser.add_argument(
        '--softargmax',
        default=False,
        action='store_true',
        help='true if using the heatmap with soft argmax model'
    )

    parser.add_argument(
        '--skeleton',
        default=True,
        action='store_true',
        help='true if using the sekeleton based model'
    )

    parser.add_argument(
        '--regression',
        default=False,
        action='store_true',
        help='true if using the sekeleton based model'
    )


def set_dataset(parser):
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='RHD_published_v2',
        help='dataset root directory'
    )

    parser.add_argument(
        '--process_training_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    parser.add_argument(
        '--process_evaluation_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

def set_model(parser):
    parser.add_argument(
        '-hgs',
        '--hg-stacks',
        default=2,
        type=int,
        metavar='N',
        help='Number of hourglasses to stack'
    )
    parser.add_argument(
        '-hgb',
        '--hg-blocks',
        default=1,
        type=int,
        metavar='N',
        help='Number of residual modules at each location in the hourglass'
    )
    parser.add_argument(
        '-nj',
        '--njoints',
        default=21,
        type=int,
        metavar='N',
        help='Number of heatmaps calsses (hand joints) to predict in the hourglass'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='whether to load checkpoint (default: none)'
    )

    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )

    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='show intermediate results'
    )

def set_training(parser):
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )

    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )

    parser.add_argument(
        '-b', '--train_batch',
        default=32,
        type=int
    )

    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1.0e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )

    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )

    parser.add_argument(
        "--lr_decay_step",
        default=50,
        type=int,
        help="Epochs after which to decay learning rate",
    )

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(
        description='PyTorch Train Hourglass On 2D Keypoint Detection')

    # HyperParameter Setting
    set_method(_parser)
    set_dataset(_parser)
    set_model(_parser)
    set_training(_parser)

    args = _parser.parse_args()

    if args.heatmap:
        train_baseline.main(args)
    elif args.softargmax:
        train_softargmax.main(args)
    elif args.regression:
        train_regression.main(args)
    elif args.skeleton:
        train_skeleton.main(args)

