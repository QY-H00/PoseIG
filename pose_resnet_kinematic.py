import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, model_urls
import copy

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RHD2Bighand_skeidx = [0, 4, 8, 12, 16, 20, 3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, 19, 18, 17]

# The GraphConv is designed based on RHD index for hand
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.ReLU(inplace=True)
        self.I = torch.eye(21).float().to(device).unsqueeze(0)
        for i in range(5):
            start = i * 4 + 1
            self.I[0, 0, start] = 1
            self.I[0, start, 0] = 1
            for j in range(3):
                self.I[0, start + j, start + 1 + j] = 1
                self.I[0, start + 1 + j, start + j] = 1
        self.I = Parameter(self.I, requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat  * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.I)
        A_hat = A_hat.repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class GraphConvBlock(nn.Module):
    def __init__(self, layers):
        super(GraphConvBlock, self).__init__()
        self.gcn_0 = GraphConv(layers[0], layers[1])
        self.gcn_1 = GraphConv(layers[1], layers[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.relu(self.gcn_0(x))
        # x = self.relu(self.gcn_1(x))
        x = self.gcn_0(x)
        x = self.gcn_1(x)
        return x


class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

class Upsampling(nn.Sequential):
    """
    3-layers deconvolution used in `Simple Baseline <https://arxiv.org/abs/1804.06208>`_.
    """

    def __init__(self, multi_channel=None, in_channel=2048, hidden_dims=(256, 256, 256), kernel_sizes=(4, 4, 4),
                 bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        if multi_channel != None:
            layers.append(nn.Sequential(*[
                nn.Conv2d(multi_channel, in_channel, kernel_size=1, stride=2, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True)]))

        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Decoder_head(nn.Module):
    """
    decoder head for uv_heatmap and 2.5d heatmap
    3 ConvTranspose2d in Unsampling
    default head has conv bn relu and conv
    """

    def __init__(self, out_features=2048, num_head_layers=2, feature_dim=256, num_keypoints=21, is_drop=False,
                 multi_channel=None):
        super(Decoder_head, self).__init__()
        self.upsampling = Upsampling(multi_channel, out_features)
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints, is_drop)

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints, is_drop=False, droprate=0.2):
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        if is_drop:
            layers.append(nn.Dropout2d(droprate))
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.upsampling(x)
        x = self.head(x)

        return x

class Decoder_mlp(nn.Module):
    def __init__(self, n_input = 2048, n_latent=128, n_keypoint=21):
        super(Decoder_mlp, self).__init__()
        self.n_keypoint = n_keypoint
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l=nn.Sequential(nn.Linear(n_input,n_latent),nn.ReLU(),
                             nn.Linear(n_latent,n_latent),nn.ReLU(),
                             nn.Linear(n_latent,n_latent),nn.ReLU(),
                             nn.Linear(n_latent, n_keypoint*2),)
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.l(x).view(x.size(0),self.n_keypoint,2)
        return x

class PoseResNet(nn.Module):
    def __init__(self, backbone, decoder, kinematic):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.gcnn = kinematic

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.gcnn(x)
        return x

class KinematicLayer(nn.Module):
    def __init__(self, layers = [64*64, 64*64, 64*64], output_layer = 64*64):
        super(KinematicLayer, self).__init__()
        self.block_0 = GraphConvBlock(layers)
        self.block_1 = GraphConvBlock(layers)
        self.block_2 = GraphConv(layers[-1], output_layer)
    
    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, 21, -1)
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x = self.block_2(x).view(bs,21,64,64) # ! this should be RHD index 
        # x = x[:,RHD2Bighand_skeidx,:] # ! convert the prediction to target index
        return x

def pose_resnet101_heatmap(pretrained = True, num_keypoints = 21, num_head_layers = 1, is_drop=False):
    backbone = resnet101(pretrained=pretrained)
    decoder = Decoder_head(2048, num_head_layers, 256, num_keypoints, is_drop=is_drop)
    kinematic = KinematicLayer()
    model = PoseResNet(backbone, decoder, kinematic)
    return model


if __name__ == '__main__':
    model = pose_resnet101_heatmap().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(1000):
        print(i)
        x = torch.randn([32, 3, 256, 256]).cuda()
        out = model(x)
        loss = torch.mean(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()