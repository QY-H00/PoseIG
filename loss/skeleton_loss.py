import torch
import torch.nn.functional as F
import data.eval_utils as eval_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkeletonLoss:
    """Computes the loss of skeleton formed of maps and kps"""

    def __init__(
            self,
            lambda_vec=1.0,
            lambda_dis=2.0,
            lambda_ske=2.0,
            lambda_kps=0.1
    ):
        self.lambda_vec = lambda_vec
        self.lambda_dis = lambda_dis
        self.lambda_ske = lambda_ske
        self.lambda_kps = lambda_kps

    def compute_loss(self, preds, targs, infos):
        # target
        targs_front_vec = targs['front_vec']  # (B, 20, res, res, 2)
        targs_back_vec = targs['back_vec']  # (B, 20, res, res, 2)
        targs_front_dis = targs['front_dis']  # (B, 20, res, res, 1)
        targs_back_dis = targs['back_dis']  # (B, 20, res, res, 1)
        targs_ske_mask = targs['ske_mask']  # (B, 20, res, res, 1)
        targs_weit_map = targs['weit_map']  # (B, 20, res, res, 1)
        targs_kps = targs['kp2d']  # (B, 21, 2)
        vis = targs['vis'][:, :, None]  # (B, 21, 1)
        bone_vis = eval_utils.generate_bone_vis(vis, device)  # (B, 20, 1)
        bone_vis = bone_vis.split(1, 1)
        batch_size = targs_kps.shape[0]

        targs_front_vec = targs_front_vec.reshape(batch_size, 20, -1).split(1, 1)
        targs_back_vec = targs_back_vec.reshape(batch_size, 20, -1).split(1, 1)
        targs_front_dis = targs_front_dis.reshape(batch_size, 20, -1).split(1, 1)
        targs_back_dis = targs_back_dis.reshape(batch_size, 20, -1).split(1, 1)
        targs_ske_mask = targs_ske_mask.reshape(batch_size, 20, -1).split(1, 1)

        # Loss Set Up
        vec_loss = torch.Tensor([0]).to(device, non_blocking=True)
        ske_loss = torch.Tensor([0]).to(device, non_blocking=True)
        dis_loss = torch.Tensor([0]).to(device, non_blocking=True)
        kps_loss = torch.Tensor([0]).to(device, non_blocking=True)

        # prediction
        for layer in range(len(preds[0])):
            front_vec = preds[0][layer]
            front_dis = preds[1][layer]
            back_vec = preds[2][layer]
            back_dis = preds[3][layer]
            ske_mask = preds[4][layer]  # (B, 20, res, res, 1)
            pred_kps = preds[5][layer]  # (B, 21, 2)

            front_vec = front_vec.reshape(front_vec.shape[0], 20, front_vec.shape[2], front_vec.shape[3], -1)
            front_dis = front_dis.reshape(front_dis.shape[0], 20, front_dis.shape[2], front_dis.shape[3], -1)
            back_vec = back_vec.reshape(back_vec.shape[0], 20, back_vec.shape[2], back_vec.shape[3], -1)
            back_dis = back_dis.reshape(back_dis.shape[0], 20, back_dis.shape[2], back_dis.shape[3], -1)
            ske_mask = ske_mask.reshape(ske_mask.shape[0], 20, ske_mask.shape[2], ske_mask.shape[3], -1)

            front_vec = targs_weit_map * front_vec
            front_dis = targs_weit_map * front_dis
            back_vec = targs_weit_map * back_vec
            back_dis = targs_weit_map * back_dis
            ske_mask = targs_weit_map * ske_mask
            front_vec = front_vec.reshape(batch_size, 20, -1).split(1, 1)
            front_dis = front_dis.reshape(batch_size, 20, -1).split(1, 1)
            back_vec = back_vec.reshape(batch_size, 20, -1).split(1, 1)
            back_dis = back_dis.reshape(batch_size, 20, -1).split(1, 1)
            ske_mask = ske_mask.reshape(batch_size, 20, -1).split(1, 1)

            for idx in range(20):

                front_veci = front_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                front_disi = front_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                back_veci = back_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                back_disi = back_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                ske_maski = ske_mask[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_front_veci = targs_front_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_front_disi = targs_front_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_back_veci = targs_back_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_back_disi = targs_back_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_ske_maski = targs_ske_mask[idx].squeeze()  # (B, 1, 4096)->(B, 4096)

                bone_visi = bone_vis[idx].squeeze()  # (B, 1, 1)->(B, 1)
                bone_visi = bone_visi[:, None]
                _vec_loss = (F.l1_loss(front_veci * bone_visi, targs_front_veci * bone_visi)
                             + F.l1_loss(back_veci * bone_visi, targs_back_veci * bone_visi)) / 2
                _dis_loss = (F.l1_loss(front_disi * bone_visi, targs_front_disi * bone_visi)
                             + F.l1_loss(back_disi * bone_visi, targs_back_disi * bone_visi)) / 2
                _ske_loss = F.l1_loss(ske_maski * bone_visi, targs_ske_maski * bone_visi)

                # _vec_loss = (F.l1_loss(front_veci, targs_front_veci)
                #              + F.l1_loss(back_veci, targs_back_veci)) / 2
                # _dis_loss = (F.l1_loss(front_disi, targs_front_disi)
                #              + F.l1_loss(back_disi, targs_back_disi)) / 2
                # _ske_loss = F.l1_loss(ske_maski, targs_ske_maski)

                vec_loss += _vec_loss * self.lambda_vec
                dis_loss += _dis_loss * self.lambda_dis
                ske_loss += _ske_loss * self.lambda_ske

            _kps_loss = eval_utils.MeanEPE(pred_kps * vis, targs_kps * vis)
            kps_loss += _kps_loss * self.lambda_kps

        vec_loss = vec_loss / len(preds[0])
        dis_loss = dis_loss / len(preds[0])
        ske_loss = ske_loss / len(preds[0])
        kps_loss = kps_loss / len(preds[0])
        loss = vec_loss + dis_loss + ske_loss + kps_loss
        map_loss = vec_loss + dis_loss + ske_loss

        return loss, vec_loss, dis_loss, ske_loss, kps_loss / self.lambda_kps, map_loss