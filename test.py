import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import data.eval_utils as eval_utils
from data.eval_utils import draw_maps
from data.eval_utils import draw_maps_with_one_bone
from data.eval_utils import draw_kps
from data.rhd_dataset import RHDDateset
from progress.bar import Bar
import numpy as np

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_pred_maps(preds, idx):
    pred_flux = preds[0][-1]
    pred_flux = pred_flux.reshape(pred_flux.shape[0], 20, pred_flux.shape[2], pred_flux.shape[3], -1)
    pred_mask = preds[1][-1]  # (B, 20, res, res, 1)
    pred_mask = pred_mask.reshape(pred_mask.shape[0], 20, pred_mask.shape[2], pred_mask.shape[3], -1)
    pred_dis = preds[2][-1]  # (B, 20, res, res, 2)
    pred_dis = pred_dis.reshape(pred_dis.shape[0], 20, pred_dis.shape[2], pred_dis.shape[3], -1)
    pred_fluxs = torch.cat((pred_flux[0], pred_mask[0]), dim=-1)
    draw_maps(pred_fluxs, pred_dis[0], idx, "pred")
    draw_maps_with_one_bone(pred_fluxs, pred_dis[0], idx, "pred")


def torch_to_numpy(image):
    img = image.clone()
    img = img.permute(1, 2, 0)
    img = img.cpu().detach().numpy()
    img = img * 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return img


def draw_pred_kps(preds, image, idx):
    pred_kps = preds[3][-1]  # (B, 21, 2)
    pred_kps = pred_kps.cpu().detach().numpy()
    pred_kps = np.ascontiguousarray(pred_kps, dtype=np.uint8)
    draw_kps(pred_kps[idx], image, idx, "pred")


def validate(val_loader, model, batch_size, stop=-1):
    am_accH = AverageMeter()

    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            preds = model(metas['img_crop'].to(device, non_blocking=True))
            pred_kps = preds[3][-1].to(device, non_blocking=True)  # (B, 21, 2)
            targ_kps = metas["uv_crop"].to(device, non_blocking=True)  # (B, 21, 2)

            val = eval_utils.MeanEPE(pred_kps, targ_kps)

            bar.suffix = (
                '({batch}/{size}) '
                'accH: {accH:.4f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                accH=val,
            )
            am_accH.update(val, batch_size)
            bar.next()
            if stop != -1 and i >= stop:
                break
        bar.finish()
        print("accH: {}".format(am_accH.avg))
    return am_accH.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Test Hourglass On 2D Keypoint Detection')

    # Test Configuration
    parser.add_argument(
        '-te',
        '--training_epochs',
        type=int,
        default=105,
        help='Number of epochs that model is trained'
    )

    parser.add_argument(
        '-idx',
        '--sample_index',
        type=int,
        default=0,
        help='The index of sample image'
    )

    parser.add_argument(
        '-batch',
        '--train_batch',
        type=int,
        default=4,
        help='The index of sample image'
    )

    parser.add_argument(
        '--test_batch',
        type=int,
        default=4,
        help='The index of sample image'
    )

    args = parser.parse_args()

    print("\nCREATE DATASET...")
    train_dataset = RHDDateset('RHD_published_v2/', 'training', input_size=256, output_full=True, aug=False)
    val_dataset = RHDDateset('RHD_published_v2/', 'evaluation', input_size=256, output_full=True, aug=False)

    print("Total train dataset size: {}".format(len(train_dataset)))
    print("Total test dataset size: {}".format(len(val_dataset)))

    print("\nLOAD DATASET...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        pin_memory=True
    )

    '''Store the cropped image and corresponding heatmap'''
    inner_index = args.sample_index % args.train_batch
    batch_index = args.sample_index // args.train_batch
    out = []
    for i, sample in enumerate(train_loader):
        if i == 0:
            vis = sample['vis21']
            hm = sample['hm']
            target = sample["uv_crop"]
            predict = eval_utils.get_heatmap_pred(hm)
            print("sample[uv_crop][0]", target[0])
            print("eval_utils.get_heatmap_pred(sample['hm'])[0]", predict[0])
            print("MeanEPE", eval_utils.MeanEPE(predict * vis[:, :, None] * 4, target * vis[:, :, None] * 4))
        else:
            break
        # for j in range(args.train_batch):
        #     sample_num = i * args.train_batch + j
        #     if sample_num >= 20:
        #         break
        #     img_crop = sample["img_crop"][j]
        #     uv_crop = sample["uv_crop"][j]
        #     hm = sample["hm"][j]
        #     out = [img_crop, uv_crop, hm]
        #     with open(f'sample/uv_crop_{sample_num}.pickle', 'wb') as f:
        #         pickle.dump(out, f)
        # if i > 20 / args.train_batch + 1:
        #     break


    '''Ground Truth Test'''
    # hm_example = []
    # kp_example = None
    # image = []
    # front_dis = []
    # front_vec = []
    # back_dis = []
    # back_vec = []
    # ske_mask = []
    # vis = []
    # dis = []
    # kps = []
    #
    # stop = -1
    # inner_index = args.sample_index % args.batch_size
    # batch_index = args.sample_index // args.batch_size
    # am_accH = AverageMeter()
    # bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    # for i, sample in enumerate(val_loader):
    #     image = sample["img_crop"]
    #     mask = sample["mask_crop"]
    #     front_vec = sample["front_vec"]
    #     front_dis = sample["front_dis"]
    #     back_vec = sample["back_vec"]
    #     back_dis = sample["back_dis"]
    #     ske_mask = sample["skeleton"]
    #     vis = sample['vis21']
    #     kps = sample['uv_crop']
    #     test_kps = eval_utils.maps_to_kp_2(front_vec.to(device), front_dis.to(device), back_vec.to(device),
    #                                        back_dis.to(device), ske_mask.to(device), device, res=64)
    #     print(test_kps.shape)
    #     val = eval_utils.MeanEPE(test_kps * vis[:, :, None].to(device), kps.to(device) * vis[:, :, None].to(device))
    #     if val > 0.1:
    #         for idx in range(args.batch_size):
    #             val_p = eval_utils.MeanEPE(test_kps[idx:idx+1] * vis[idx:idx+1, :, None].to(device), kps[idx:idx+1].to(device) * vis[idx:idx+1, :, None].to(device))
    #             if val_p > 0.1:
    #                 print(i*args.batch_size+idx, val_p)
    #
    #     bar.suffix = (
    #         '({batch}/{size}) '
    #         'accH: {accH:.4f} | '
    #     ).format(
    #         batch=i + 1,
    #         size=len(val_loader),
    #         accH=val,
    #     )
    #     am_accH.update(val, args.batch_size)
    #     bar.next()
    #     if stop != -1 and i >= stop:
    #         break
    # bar.finish()
    # print("accH: {}".format(am_accH.avg))

    '''Check correspondence of a certain sample'''
    # hm_example = []
    # kp_example = None
    # image = []
    # front_dis = []
    # front_vec = []
    # back_dis = []
    # back_vec = []
    # ske_mask = []
    # weit_map = []
    # vis = []
    # dis = []
    # kps = []
    #
    # stop = -1
    # inner_index = args.sample_index % args.batch_size
    # batch_index = args.sample_index // args.batch_size
    # am_accH = AverageMeter()
    # for i, sample in enumerate(val_loader):
    #     if i == batch_index:
    #         image = sample["img_crop"]
    #         mask = sample["mask_crop"]
    #         front_vec = sample["front_vec"]
    #         front_dis = sample["front_dis"]
    #         back_vec = sample["back_vec"]
    #         back_dis = sample["back_dis"]
    #         ske_mask = sample["skeleton"]
    #         weit_map = sample["weit_map"]
    #         vis = sample['vis21']
    #         kps = sample['uv_crop']
    #
    #
    # img = torch_to_numpy(image[inner_index])
    # test_kps = eval_utils.maps_to_kp_2(front_vec.to(device), front_dis.to(device), back_vec.to(device), back_dis.to(device), ske_mask.to(device), device, res=64)
    # test_kps = test_kps[inner_index].cpu().detach().numpy()
    # front_vec = front_vec[inner_index].cpu().detach().numpy()
    # front_dis = front_dis[inner_index].cpu().detach().numpy()
    # back_vec = back_vec[inner_index].cpu().detach().numpy()
    # back_dis = back_dis[inner_index].cpu().detach().numpy()
    # ske_mask = ske_mask[inner_index].cpu().detach().numpy()
    # weit_map = weit_map[inner_index].cpu().detach().numpy()
    # # vis = vis[inner_index].cpu().detach().numpy()
    # # print("test_kps")
    # # print(test_kps)
    # # val_p = eval_utils.MeanEPE(test_kps[None, :, :] * vis[None, :, None].to(device),
    # #                            kps[inner_index][None, :, :].to(device) * vis[None, :, None].to(device))
    # # print("targ_kps")
    # # print(kps[inner_index])
    # # print("accH: ", val_p)
    # # print(test_kps.shape)
    # test_kps = np.ascontiguousarray(test_kps, dtype=np.uint8)
    # draw_kps(test_kps, img, args.sample_index, "targ")
    # draw_maps_2(front_vec, front_dis, back_vec, back_dis, ske_mask, weit_map, args.sample_index, "targ")
    # draw_maps_with_one_bone_2(front_vec, front_dis, back_vec, back_dis, ske_mask, weit_map, args.sample_index, "targ")

    '''Validation maps of a trained example'''
    # hm_example = []
    # kp_example = None
    # image = []
    # front_dis = []
    # front_vec = []
    # back_dis = []
    # back_vec = []
    # ske_mask = []
    # weit_map = []
    # vis = []
    # dis = []
    # kps = []
    #
    # stop = -1
    # inner_index = args.sample_index % args.batch_size
    # batch_index = args.sample_index // args.batch_size
    # am_accH = AverageMeter()
    # for i, sample in enumerate(val_loader):
    #     if i == batch_index:
    #         image = sample["img_crop"]
    #         mask = sample["mask_crop"]
    #         front_vec = sample["front_vec"]
    #         front_dis = sample["front_dis"]
    #         back_vec = sample["back_vec"]
    #         back_dis = sample["back_dis"]
    #         ske_mask = sample["skeleton"]
    #         weit_map = sample["weit_map"]
    #         vis = sample['vis21']
    #         targ_kps = sample['uv_crop']
    #         break
    # model = torch.load(f"trained_model_v1.3/skeleton_model_after_{args.training_epochs}_epochs.pkl")
    # model = model.to(device)
    # preds = model(image)
    # pred_front_vec = preds[0][-1][inner_index].cpu().detach().numpy()
    # pred_front_vec = pred_front_vec.reshape(20, front_vec.shape[2], front_vec.shape[3], -1)
    # pred_front_dis = preds[1][-1][inner_index].cpu().detach().numpy()
    # pred_front_dis = pred_front_dis.reshape(20, front_dis.shape[2], front_dis.shape[3], -1)
    # pred_back_vec = preds[2][-1][inner_index].cpu().detach().numpy()
    # pred_back_vec = pred_back_vec.reshape(20, back_vec.shape[2], back_vec.shape[3], -1)
    # pred_back_dis = preds[3][-1][inner_index].cpu().detach().numpy()
    # pred_back_dis = pred_back_dis.reshape(20, back_dis.shape[2], back_dis.shape[3], -1)
    # pred_ske_mask = preds[4][-1][inner_index].cpu().detach().numpy()
    # pred_ske_mask = pred_ske_mask.reshape(20, ske_mask.shape[2], ske_mask.shape[3], -1)
    # pred_kps = preds[5][-1][inner_index]
    # weit_map = weit_map.cpu().detach().numpy()[inner_index]
    # image = torch_to_numpy(image[inner_index])
    # pred_kps = np.ascontiguousarray(pred_kps.cpu().detach().numpy(), dtype=np.uint8)
    # front_vec = front_vec[inner_index].cpu().detach().numpy()
    # print("pred_front_vec", np.mean(front_vec - pred_front_vec))
    # print("average pred vec", np.mean(pred_front_vec[0], axis=(0, 1)))
    # print("average targ vec", np.mean(front_vec[0], axis=(0, 1)))
    # draw_kps(pred_kps, image, args.sample_index, "pred")
    # draw_maps_2(pred_front_vec, pred_front_dis, pred_back_vec, pred_back_dis, pred_ske_mask, weit_map, args.sample_index, "pred")
    # draw_maps_with_one_bone_2(pred_front_vec, pred_front_dis, pred_back_vec, pred_back_dis, pred_ske_mask, weit_map, args.sample_index, "pred")

    '''Check the function kp_to_maps_2'''
    # hm_example = []
    # kp_example = None
    # image = []
    # targ_kps = []
    # front_dis = []
    # front_vec = []
    # back_dis = []
    # back_vec = []
    #
    # stop = -1
    # inner_index = args.sample_index % args.batch_size
    # batch_index = args.sample_index // args.batch_size
    # am_accH = AverageMeter()
    # for i, sample in enumerate(val_loader):
    #     if i == batch_index:
    #         image = sample["img_crop"]
    #         targ_kps = sample['uv_crop']
    #         break
    # kp = targ_kps[inner_index] / 4
    # kp = kp.cpu().detach().numpy()
    # front_vec, front_dis, back_vec, back_dis, ske_mask, weit_map = eval_utils.kp_to_maps_2(kp, res=64, sigma=1)
    # image = image[inner_index]
    # img = torch_to_numpy(image)
    # test_kps = eval_utils.maps_to_kp_2(torch.from_numpy(front_vec[None,:]).to(device), torch.from_numpy(front_dis[None,:]).to(device), torch.from_numpy(back_vec[None,:]).to(device),
    #                                    torch.from_numpy(back_dis[None,:]).to(device), torch.from_numpy(ske_mask[None,:]).to(device), device, res=64)
    # test_kps = np.ascontiguousarray(test_kps.cpu().detach().numpy(), dtype=np.uint8)
    # test_kps = test_kps[0]
    # draw_kps(test_kps, img, args.sample_index, "gaussian_regression")
    # print("test_kps")
    # print(test_kps)
    # print("targ_kps")
    # print(targ_kps[inner_index])
    # draw_maps_2(front_vec, front_dis, back_vec, back_dis, ske_mask, weit_map, args.sample_index, "gaussian_mask")
    # draw_maps_with_one_bone_2(front_vec, front_dis, back_vec, back_dis, ske_mask, weit_map,
    #                           args.sample_index, "gaussian_mask")
