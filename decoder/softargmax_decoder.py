import torch
import torch.nn as nn
from kornia.geometry.dsnt import spatial_softmax_2d, spatial_softargmax_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftargmaxDecoder(nn.Module):
    def __init__(self, is_stacking=False):
        super(SoftargmaxDecoder, self).__init__()
        self.beta = torch.ones(21, device=device, requires_grad=True)
        self.is_stacking = is_stacking

    def regress25d(self, heatmaps, beta):
        bs = heatmaps.shape[0]
        betas = beta.clone().view(1, 21, 1).repeat(bs, 1, 1)
        uv_heatmaps = spatial_softmax_2d(heatmaps, betas)
        coord_out = spatial_softargmax_2d(uv_heatmaps, normalized_coordinates=False)
        coord_out_final = coord_out.clone()

        return coord_out_final.view(bs, 21, 2)

    def forward(self, x):
        if self.is_stacking:
            stacks = len(x)
            res = list(range(stacks))
            for i in range(stacks):
                res[i] = self.regress25d(x[0][i], self.beta)
            return res
        else:
            res = self.regress25d(x, self.beta)
            return res
