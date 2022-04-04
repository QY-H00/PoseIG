import torch
import data.eval_utils as eval_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KeypointLoss:
    def compute_loss(self, preds, targs, infos, is_stacking=False):
        vis = infos['vis'][:, :, None]
        kp_loss = torch.Tensor([0]).to(device, non_blocking=True)
        if is_stacking:
            for pred in preds:
                pred_kps = pred.reshape(pred.shape[0], 21, 2)
                targ_kps = targs['kp2d']  # targ_kps resolution is 256 * 256
                kp_loss += eval_utils.MeanEPE(pred_kps * vis * 4, targ_kps * vis)
            return kp_loss / len(preds)
        else:
            pred_kps = preds.reshape(preds.shape[0], 21, 2)
            targ_kps = targs['kp2d']  # targ_kps resolution is 256 * 256
            kp_loss = eval_utils.MeanEPE(pred_kps * vis * 4, targ_kps * vis)
            return kp_loss
