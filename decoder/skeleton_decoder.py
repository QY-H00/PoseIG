import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkeletonDecoder(nn.Module):
    def __init__(self):
        super(SkeletonDecoder, self).__init__()

    def split(self, x):
        front_vecs, front_diss, back_vecs, back_diss, maskss = torch.split(x, [40, 20, 40, 20, 20], dim=1)
        return front_vecs, front_diss, back_vecs, back_diss, maskss

    def maps_to_kps(self, front_vecs, front_diss, back_vecs, back_diss, maskss, res=64):
        batch_size = front_vecs.shape[0]
        locs = torch.zeros([res, res, 2]).to(device, non_blocking=True)
        locs[:, :, 0], locs[:, :, 1] = torch.meshgrid(torch.arange(0, res), torch.arange(0, res))
        kps = torch.zeros([batch_size, 21, 2]).to(device, non_blocking=True)

        front_vecs = front_vecs.reshape(batch_size, 20, res, res, 2)
        front_diss = front_diss.reshape(batch_size, 20, res, res, 1)
        back_vecs = back_vecs.reshape(batch_size, 20, res, res, 2)
        back_diss = back_diss.reshape(batch_size, 20, res, res, 1)
        maskss = maskss.reshape(batch_size, 20, res, res, 1)

        # predicts the wrist keypoint 0
        for i in range(5):
            back_vec = back_vecs[:, 4 * i, :, :, :]
            back_dis = back_diss[:, 4 * i, :, :, :]
            front_vec = front_vecs[:, 4 * i, :, :, :]
            front_dis = front_diss[:, 4 * i, :, :, :]
            masks = maskss[:, 4 * i, :, :, :]
            j = 0
            for mask in masks:
                if torch.sum(mask) != 0:
                    # proposal = mask * (back_vec[j] * back_dis[j, :, :, 0, None] * res + locs)
                    proposal = mask * (-front_vec[j] * front_dis[j, :, :, 0, None] * res + locs)
                    kps[j][0] += torch.sum(torch.sum(proposal, dim=0), dim=0) / (torch.sum(mask) + 1e-6)
                    if torch.sum(mask) == 0:
                        print("flag")
                j = j + 1
        kps[:, 0] = kps[:, 0] / 5

        for i in range(5):
            for j in range(4):
                k = 0
                for masks in maskss:
                    front_votes = masks[4 * i + j] * (
                            front_vecs[k][4 * i + j] * front_diss[k, 4 * i + j, :, :, 0, None] * res + locs)
                    front_pred = front_votes / (torch.sum(masks[4 * i + j]) + 1e-6)
                    kps[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred, dim=0), dim=0)
                    # if j != 3:
                    #     # predict keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                    #     back_votes = masks[4 * i + j + 1] * (
                    #             back_vecs[k][4 * i + j + 1] * back_diss[k, 4 * i + j + 1, :, :, 0, None] * res + locs)
                    #     back_pred = back_votes / (torch.sum(masks[4 * i + j + 1]) + 1e-6)
                    #     kps[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred + back_pred, dim=0), dim=0) / 2
                    # else:
                    #     # predicts the location of fingertips 1, 5, 9, 13, 17
                    #     kps[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred, dim=0), dim=0)
                    k = k + 1

        return kps

    def forward(self, x):
        stacks = len(x)
        front_vec = list(range(stacks))
        front_dis = list(range(stacks))
        back_vec = list(range(stacks))
        back_dis = list(range(stacks))
        ske_mask = list(range(stacks))
        kps = list(range(stacks))
        for i in range(stacks):
            front_vec[i], front_dis[i], back_vec[i], back_dis[i], ske_mask[i] = self.split(x[0][i])
            kps[i] = self.maps_to_kps(front_vec[i], front_dis[i], back_vec[i], back_dis[i], ske_mask[i])
        return front_vec, front_dis, back_vec, back_dis, ske_mask, kps
