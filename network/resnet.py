import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, pretrained):
        super(ResNet, self).__init__()
        self.resnet152 = models.resnet152(pretrained=pretrained)

    def forward(self, x):
        return self.resnet152(x)
