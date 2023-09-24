import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ASSP Encoder structure

    # 1 = 1x1 convolution → BatchNorm → ReLu
    # 2 = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
    # 3 = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
    # 4 = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
    # 5 = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
    # 6 = concatenate(1 + 2 + 3 + 4 + 5)
    # 7 = 1x1 convolution → BatchNorm → ReLu


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ASPP, self).__init__()
        
        # 1 x 1 convolution -> batchnorm -> relu
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        
        # 3 x 3 convolution w/ rate = 6 -> batchnorm ->relu
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 6, dilation = 6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
        
        # 3 x 3 convolution w/ rate = 12 -> batchnorm -> relu
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 12, dilation = 12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
        
        # 3 x 3 convolution w/ rate = 18 -> batchnorm -> relu
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 18, dilation = 18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)
        
        # AdaptiveAvgPool2d -> 1 x 1 convolution -> batchnorm -> relu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size = 1) # 256 * 5 = 1280
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size = 1)
        
    def forward(self, feature_map):
        # feature map 의 shape 은 (batch_size, in_channels, height / output_stride, width / output_stride)
        feature_map_h = feature_map.size()[2] # (== h / 16)
        feature_map_w = feature_map.size()[3] # (== w / 16)
        
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        
        # shape: (batch_size, in_channels, 1, 1)
        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size = (feature_map_h, feature_map_w), mode = 'bilinear')
        
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)
        
        return out