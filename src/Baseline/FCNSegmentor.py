import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torchinfo
import torch.nn.functional as F

class FCNSegmentor0(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet18.children())[:-4]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False
        self.feature_extractor = resnet18

        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=256)

        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)

        self.deconv3 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)

        self.deconv4 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)

        self.act_func = torch.nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(input)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        #output = torch.sigmoid(x)  
        output = x
        return output

class FCNSegmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet18.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)

        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=256)

        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)

        self.deconv3 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)

        self.deconv4 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)

        self.act_func = torch.nn.ReLU()

    def forward(self, input):
        x = self.feature_extractor(input)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        #output = torch.sigmoid(x)
        output = x
        return output

