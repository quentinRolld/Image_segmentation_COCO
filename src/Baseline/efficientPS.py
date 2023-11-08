import torch
import torch.nn as nn
import torchvision.models as models

#1st requirement SSH

class LSFEComponent(nn.Module):
    def __init__(self, in_channels):
        super(LSFEComponent, self).__init__()
        
        # First 3x3 depthwise separable convolution with 128 output filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Second 3x3 depthwise separable convolution with 128 output filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply the first 3x3 depthwise separable convolution
        x = self.conv1(x)
        
        # Apply the second 3x3 depthwise separable convolution
        x = self.conv2(x)
        
        return x

#2nd requirement SSH

class DPCModule(nn.Module):
    def __init__(self, in_channels=2, out_channels=256):
        super(DPCModule, self).__init__()
        
        # 3x3 depthwise separable convolution with dilation rate (1,6)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=6, dilation=6, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=10, dilation=10, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=15, dilation=15, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 1x1 convolution to concatenate branches
        self.conv_cat = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_cat = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # Apply the main convolution
        x_main = self.conv1(x)
        print(x_main.shape)
        
        # Apply parallel branches
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x_main)  # Takes the output of conv1 as input
        print(branch1.shape)
        print(branch2.shape)
        print(branch3.shape)
        print(branch4.shape)
        # Concatenate the branches
        concatenated = torch.cat([branch1, branch2, branch3, branch4, x_main], dim=1)
        
        # Apply 1x1 convolution to concatenate branches
        x = self.conv_cat(concatenated)
        x = self.bn_cat(x)
        x = self.relu(x)
        
        return x

# SSH 3rd requirement

class MismatchCorrectionModule(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(MismatchCorrectionModule, self).__init__()
        
        # Cascaded 3x3 depthwise separable convolutions with 128 output channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Bilinear upsampling layer to upsample feature maps by a factor of 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Apply cascaded convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Upsample the feature maps
        x = self.upsample(x)
        
        return x


class InstanceSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InstanceSegmentationHead, self).__init__()
        
        # Replace standard convolutions with depthwise separable convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Classification convolution for predicting class scores
        self.classification_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Mask prediction convolution for predicting instance masks
        self.mask_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply depthwise separable convolutions
        x = self.conv1(x)
        
        # Predict class scores
        class_scores = self.classification_conv(x)
        
        # Predict instance masks
        masks = self.mask_conv(x)
        
        return class_scores, masks


class EfficientPS(nn.Module):
    def __init__(self):
        super(EfficientPS, self).__init__()
        # Charger ResNet18 comme backbone et modifier le nombre de canaux d'entrée
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-4]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False

        # Backbone (ResNet18 modifié)
        self.feature_extractor = resnet18
        
        
        # Couches pour la segmentation sémantique
        self.lsfe_component = LSFEComponent(in_channels=256)  # Ajuster les canaux d'entrée selon votre cas d'utilisation
        self.dpc_module = DPCModule(in_channels=256)  # Ajuster les canaux d'entrée selon votre cas d'utilisation
        self.mismatch_correction_module = MismatchCorrectionModule(in_channels=256)  # Ajuster les canaux d'entrée selon votre cas d'utilisation

        # Couche de classification pour la segmentation sémantique
        self.semantic_segmentation_classifier = nn.Conv2d(256, 2, kernel_size=1, stride=1)

    def forward(self, input):
        # Extraction des caractéristiques avec le backbone (ResNet18 modifié)
        x = self.feature_extractor(input)
        
        # Couches additionnelles pour la segmentation sémantique
        semantic_features = self.lsfe_component(x)
        semantic_features = self.dpc_module(semantic_features)
        semantic_features = self.mismatch_correction_module(semantic_features)
        
        # Prédiction de la segmentation sémantique
        semantic_segmentation_output = self.semantic_segmentation_classifier(semantic_features)
        
        return semantic_segmentation_output







