import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import torch
import argparse
import torch.backends.cudnn as cudnn
import data.eval_utils as eval_utils
from data.rhd_dataset import RHDDateset
from data.eval_utils import AverageMeter
from loss.keypoint_loss import KeypointLoss
from loss.heatmap_loss import HeatmapLoss
from progress.bar import Bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def one_forward_pass(sample, model, criterion, args, is_training=True):
    """ prepare target """
    img = sample['img_crop'].to(device, non_blocking=True)
    kp2d = sample['uv_crop'].to(device, non_blocking=True)
    vis = sample['vis21'].to(device, non_blocking=True)

    ''' heatmap generation '''
    # need to modify
    hm = sample['hm'].to(device, non_blocking=True)

    ''' prepare infos '''
    # need to modify
    hm_veil = sample['hm_veil'].to(device, non_blocking=True)
    infos = {
        'hm_veil': hm_veil,
        'batch_size': args.test_batch,
        'vis': vis
    }

    targets = {
        'clr': img,
        'hm': hm,
        'kp2d': kp2d,
        'vis': vis
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(img)
    ''' ----------------  Forward End   ---------------- '''

    loss = torch.Tensor([0]).cuda()
    if not is_training:
        return results, {**targets, **infos}, loss

    ''' compute losses '''
    loss = criterion.compute_loss(results, targets, infos)

    return results, {**targets, **infos}, loss


def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    am_accH = AverageMeter()

    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, loss = one_forward_pass(
                metas, model, criterion, args, is_training=False
            )
            pred_kps = results
            targ_kps = targets['kp2d']
            vis = targets['vis'][:, :, None]
            val = 0
            if isinstance(criterion, KeypointLoss):
                val = eval_utils.MeanEPE(pred_kps * vis * 4, targ_kps * vis)
            else:
                val = eval_utils.accuracy_heatmap(
                    results,
                    targets['hm'],
                    targets['vis'][:, :, None]
                )
            bar.suffix = (
                '({batch}/{size}) '
                'accH: {accH:.4f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                accH=val,
            )
            am_accH.update(val, args.test_batch)
            bar.next()
            if stop != -1 and i >= stop:
                break
        bar.finish()
        print("accH: {}".format(am_accH.avg))
    return am_accH.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='PyTorch Train Hourglass On 2D Keypoint Detection')

    parser.add_argument(
        '--mode',
        type=str,
        default='regression',
        help='model used'
    )

    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int
    )

    parser.add_argument(
        '-j', '--workers',
        default=2,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )

    parser.add_argument(
        '--sigma',
        default=3,
        type=int,
        help='detection-based parameters'
    )

    args = parser.parse_args()
    print("\nCREATE DATASET...")
    val_dataset = RHDDateset('RHD_published_v2/', 'evaluation', input_size=256, output_full=True, aug=False)
    print("\nLOAD DATASET...")
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )

    model = torch.load(f'trained_model_{args.mode}/model.pkl')

    if args.mode == "baseline":
        model = torch.load(f'trained_model_{args.mode}/model_sigma{args.sigma}_after_100_epochs.pkl')

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    criterion = KeypointLoss() if args.mode != "baseline" else HeatmapLoss()
    val_indicator = validate(val_loader, model, criterion, args=args)
    print("Finish validation.")
