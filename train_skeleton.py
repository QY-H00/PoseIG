import argparse
import time
import os
import os.path as osp

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim

from datetime import datetime
from tensorboardX import SummaryWriter
from progress.bar import Bar
from termcolor import cprint

import process_data
from data import eval_utils
from data.eval_utils import AverageMeter
from data.RHD import RHD_DataReader_With_File

from network.hourglass import NetStackedHourglass
from decoder.skeleton_decoder import SkeletonDecoder
from loss.skeleton_loss import SkeletonLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main(args):
    """Main process"""

    '''Set up the network'''

    print("\nCREATE NETWORK")
    encoder = NetStackedHourglass(nclasses=140)
    decoder = SkeletonDecoder()
    model = nn.Sequential(encoder, decoder)
    model = nn.DataParallel(model)

    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    criterion = SkeletonLoss()
    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            },
        ],
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    '''Generate dataset'''

    print("\nCREATE DATASET...")
    # Create evaluation dataset
    if args.process_evaluation_data:
        process_data.process_evaluation_data(args)

    eval_dataset = RHD_DataReader_With_File(mode="evaluation", path="data_v2.0")

    val_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print("Total test dataset size: {}".format(len(eval_dataset)))

    # Create training dataset
    if args.process_training_data:
        process_data.process_training_data(args)

    last = time.time()
    train_dataset = RHD_DataReader_With_File(mode="training", path="data_v2.0")
    print("loading training dataset time", time.time() - last)

    train_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print("Total train dataset size: {}".format(len(train_dataset)))

    '''Set up the monitor'''

    loss_log_dir = osp.join('tensorboard', f'loss_{datetime.now().strftime("%Y%m%d_%H%M")}')
    loss_writer = SummaryWriter(log_dir=loss_log_dir)
    error_log_dir = osp.join('tensorboard', f'error_{datetime.now().strftime("%Y%m%d_%H%M")}')
    error_writer = SummaryWriter(log_dir=error_log_dir)

    '''Start Training'''

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])

        loss_avg = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
            epoch=epoch
        )

        # Validate the correctness every 5 epochs
        if epoch % 5 == 4:
            train_indicator = validate(train_loader, model, criterion, args=args)
            val_indicator = validate(val_loader, model, criterion, args=args)
            print(f'Save skeleton_model.pkl after {epoch + 1} epochs')
            error_writer.add_scalar('Validation Indicator', val_indicator, epoch)
            error_writer.add_scalar('Training Indicator', train_indicator, epoch)
            torch.save(model, f'trained_model_v1.6/skeleton_model_after_{epoch + 1}_epochs.pkl')

        # Draw the loss curve and validation indicator curve
        loss_writer.add_scalar('Loss', loss_avg, epoch)

        scheduler.step()

    '''Save Model'''

    print("Save skeleton_model.pkl after total training")
    torch.save(model, 'trained_model_v1.6/skeleton_model.pkl')
    cprint('All Done', 'yellow', attrs=['bold'])

    return 0  # end of main


def one_forward_pass(sample, model, criterion, args, is_training=True):
    # forward pass the sample into the model and compute the loss of corresponding sample
    """ prepare target """
    img = sample['img_crop'].to(device, non_blocking=True)
    kp2d = sample['uv_crop'].to(device, non_blocking=True)
    vis = sample['vis21'].to(device, non_blocking=True)

    ''' skeleton map generation '''
    front_vec = sample['front_vec'].to(device, non_blocking=True)
    front_dis = sample['front_dis'].to(device, non_blocking=True)
    back_vec = sample['back_vec'].to(device, non_blocking=True)
    back_dis = sample['back_dis'].to(device, non_blocking=True)
    ske_mask = sample['skeleton'].to(device, non_blocking=True)
    weit_map = sample['weit_map'].to(device, non_blocking=True)

    ''' prepare infos '''
    infos = {
        'batch_size': args.train_batch
    }

    targets = {
        'clr': img,
        'front_vec': front_vec,
        'front_dis': front_dis,
        'back_vec': back_vec,
        'back_dis': back_dis,
        'ske_mask': ske_mask,
        'kp2d': kp2d,
        'vis': vis,
        'weit_map': weit_map
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(img)
    ''' ----------------  Forward End   ---------------- '''

    loss = torch.Tensor([0]).cuda()
    if not is_training:
        return results, {**targets, **infos}, loss
    else:
        ''' compute losses '''
        loss = criterion.compute_loss(results, targets, infos)
        return results, {**targets, **infos}, loss


def train(train_loader, model, criterion, optimizer, args, epoch):
    """Train process"""
    '''Set up configuration'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss = AverageMeter()
    vec_loss = AverageMeter()
    dis_loss = AverageMeter()
    ske_loss = AverageMeter()
    kps_loss = AverageMeter()
    last = time.time()
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))

    '''Start Training'''
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, loss = one_forward_pass(
            sample, model, criterion, args, is_training=True
        )

        '''Update the loss after each sample'''

        am_loss.update(
            loss[0].item(), targets['batch_size']
        )
        vec_loss.update(
            loss[1].item(), targets['batch_size']
        )
        dis_loss.update(
            loss[2].item(), targets['batch_size']
        )
        ske_loss.update(
            loss[3].item(), targets['batch_size']
        )
        kps_loss.update(
            loss[4].item(), targets['batch_size']
        )

        ''' backward and step '''
        optimizer.zero_grad()
        # loss[1].backward()
        if epoch < 60:
            loss[5].backward()
        else:
            loss[0].backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'l: {loss:.5f} | '
            'lV: {lossV:.5f} | '
            'lD: {lossD:.5f} | '
            'lM: {lossM:.5f} | '
            'lK: {lossK:.5f} | '
        ).format(
            batch=i + 1,
            size=len(train_loader),
            loss=am_loss.avg,
            lossV=vec_loss.avg,
            lossD=dis_loss.avg,
            lossM=ske_loss.avg,
            lossK=kps_loss.avg
        )
        bar.next()
    bar.finish()
    return am_loss.avg


def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    am_accH = AverageMeter()

    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, loss = one_forward_pass(
                metas, model, criterion, args=args, is_training=False
            )
            pred_kps = results[5][-1]  # (B, 21, 2)
            targ_kps = targets["kp2d"]  # (B, 21, 2)
            vis = targets['vis'][:, :, None]

            val = eval_utils.MeanEPE(pred_kps * vis, targ_kps * vis)
            #             print(targ_kps[idx])
            bar.suffix = (
                '({batch}/{size}) '
                'accH: {accH:.4f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                accH=val,
            )
            am_accH.update(val, args.train_batch)
            bar.next()
            if stop != -1 and i >= stop:
                break
        bar.finish()
        print("accH: {}".format(am_accH.avg))
    return am_accH.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train Hourglass On 2D Keypoint Detection')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='RHD_published_v2',
        help='dataset root directory'
    )

    # Dataset setting
    parser.add_argument(
        '--process_training_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    # Dataset setting
    parser.add_argument(
        '--process_evaluation_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    # Model Structure
    # hourglass:
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

    # Dataset setting
    parser.add_argument(
        '-cc',
        '--checking_cycle',
        type=int,
        default=5,
        help='How many batches to save the model at once'
    )

    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=119,
        type=int,
        metavar='N',
        help='number of total epochs to run'
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
        type=int,
        metavar='N',
        help='train batch size'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int,
        metavar='N',
        help='test batch size'
    )

    parser.add_argument(
        '-lr', '--learning_rate',
        default=1.0e-3,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=50,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed'],
        type=str,
        help="sub modules contained in model"
    )

    main(parser.parse_args())
