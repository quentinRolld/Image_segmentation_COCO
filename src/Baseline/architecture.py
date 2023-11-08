import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torchinfo
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Segmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet18.children())[:-4]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False
        self.feature_extractor = resnet18
        self.conv1 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=128)
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)
        self.act_func = torch.nn.ReLU6()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output = x
        return output
        
class Segmentor_efficientnet(nn.Module): #nouveau segementor avec efficient net au lieu de resnet
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        modules = list(efficientnet.children())[:-4]
        efficientnet = nn.Sequential(*modules)
        for p in efficientnet.parameters():
            p.requires_grad = False
        self.feature_extractor = efficientnet
        self.conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding=3,
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.act_func = nn.ReLU6()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print("Input Shape:", input.shape)  # Print the input shape
        x = self.feature_extractor(input)
        print("Feature Extractor Output Shape:", x.shape)  # Print the output shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output = x
        return output
        
class Segmentor_mobilenet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet_v2 = models.MobileNetV2()
        modules = list(mobilenet_v2.children())[:-1]
        mobilenet_v2 = nn.Sequential(*modules)
        for p in mobilenet_v2.parameters():
            p.requires_grad = False
        self.feature_extractor = mobilenet_v2
        self.conv1 = torch.nn.Conv2d(
            in_channels=1280,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=128)
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding=3,
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
        )

        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0,
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)
        self.act_func = torch.nn.ReLU6()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output = x
        return output
       
class Segmentor_DeepLabV3(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Utilisez le modèle DeepLabV3 avec un backbone ResNet
        #self.model = models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1', progress=True)
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='COCO_WITH_VOC_LABELS_V1', progress=True)
        self.model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)['out'] 
    
    
''' Cette classe définit un modèle U-Net pour la segmentation d'image. Le modèle prend en entrée une image et renvoie une carte de segmentation binaire en sortie.

Le modèle est composé d'un encodeur et d'un décodeur. L'encodeur est constitué de quatre couches de convolution avec des couches de normalisation. Le décodeur est constitué de trois couches de convolution transposée avec des couches de normalisation.

La méthode forward définit comment les données sont propagées à travers le réseau. Elle prend une entrée de type torch.Tensor et renvoie une sortie de type torch.Tensor.

 '''

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Use ResNet18 as the encoder with pretrained weights
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet18.children())[:-4]  # remove last layers
        self.encoder = nn.Sequential(*modules)

        # Freeze the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder part of the U-Net architecture
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128, 64)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(96, 32)
        self.upconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x):
       # Encoder part
        x1 = self.encoder[0:4](x)  # size=(N, 64, x.H/2, x.W/2)
        x2 = self.encoder[4:5](x1)  # size=(N, 64, x.H/4, x.W/4)
        x3 = self.encoder[5:6](x2)  # size=(N, 128, x.H/8, x.W/8)
        x4 = self.encoder[6:](x3)  # size=(N, 256, x.H/16, x.W/16)

        # Decoder part
        x = self.upconv1(x4)  # size=(N, 128, x.H/8, x.W/8)
        x = torch.cat([x, F.interpolate(x3, size=x.size()[2:])], dim=1)
        x = self.decoder1(x)  # size=(N, 128, x.H/8, x.W/8)

        x = self.upconv2(x)  # size=(N, 64, x.H/4, x.W/4)
        x = torch.cat([x, F.interpolate(x2, size=x.size()[2:])], dim=1)
        x = self.decoder2(x)  # size=(N, 64, x.H/4, x.W/4)

        x = self.upconv3(x)  # size=(N, 32, x.H/2, x.W/2)
        x = torch.cat([x, F.interpolate(x1, size=x.size()[2:])], dim=1)
        x = self.decoder3(x)  # size=(N, 32, x.H/2, x.W/2)

        x = self.upconv4(x)  # size=(N, 1, x.H, x.W)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    

    