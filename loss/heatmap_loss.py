import torch
import torch.nn.functional as F


class HeatmapLoss:
    def __init__(
            self,
            lambda_hm=1.0,
            lambda_mask=1.0,
            lambda_joint=1.0,
            lambda_dep=1.0,
    ):
        self.lambda_dep = lambda_dep
        self.lambda_hm = lambda_hm
        self.lambda_joint = lambda_joint
        self.lambda_mask = lambda_mask

    def compute_loss(self, preds, targs, infos, is_stacking=False):

        hm_veil = infos['vis']  # orginally: infos['hm_veil']
        batch_size = infos['batch_size']

        # compute hm_loss anyway
        hm_loss = torch.Tensor([0]).cuda()
        hm_veil = hm_veil.unsqueeze(-1)
        if is_stacking:
            for pred_hm in preds[0]:
                njoints = pred_hm.size(1)
                batch_size = pred_hm.shape[0]
                pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1, 1)
                targ_hm = targs['hm'].reshape((batch_size, njoints, -1)).split(1, 1)
                for idx in range(njoints):
                    pred_hmi = pred_hm[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                    targ_hmi = targ_hm[idx].squeeze()
                    hm_loss += 0.5 * F.mse_loss(
                        pred_hmi.mul(hm_veil[:, idx]),  # (B, 4096) mul (B, 1)
                        targ_hmi.mul(hm_veil[:, idx])
                        )
        else:
            njoints = preds.shape[1]
            batch_size = preds.shape[0]
            pred_hm = preds.reshape((batch_size, njoints, -1)).split(1, 1)
            targ_hm = targs['hm'].reshape((batch_size, njoints, -1)).split(1, 1)
            for idx in range(njoints):
                pred_hmi = pred_hm[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targ_hmi = targ_hm[idx].squeeze()
                hm_loss += 0.5 * F.mse_loss(
                    pred_hmi.mul(hm_veil[:, idx]),  # (B, 4096) mul (B, 1)
                    targ_hmi.mul(hm_veil[:, idx])
                )
        return hm_loss
