from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork
from util.CBAM import ChannelAttention, SpatialAttention


class DualResNet50(nn.Module):
    """相机、声呐各一个 ResNet-50，分别提取 C3 / C4 / C5，按通道拼接后送入 FPN。

    """

    def __init__(self, out_channels=256):
        super().__init__()        
        self.camera_ca = nn.ModuleList(ChannelAttention(512),
        ChannelAttention(1024),
        ChannelAttention(2048))
        self.camera_sa = nn.ModuleList(SpatialAttention(), SpatialAttention(), SpatialAttention())
        #each layer need its own weight to extarct the feature

        self.sonar_ca = nn.ModuleList(ChannelAttention(512),
                ChannelAttention(1024),
                ChannelAttention(2048))
        self.sonar_sa = nn.ModuleList(SpatialAttention(), SpatialAttention(), SpatialAttention())


        self.cam = resnet50()
        self.sonar = resnet50()
        in_channels_list = [1024, 2048, 4096]

    def features(self, x, y):
        """用某一个 ResNet-50 提取 C3 / C4 / C5，返回字典 {'0': C3, '1': C4, '2': C5}。"""
        net_cam = self.cam
        net_son = self.sonar
        # stem：ResNet 的第一部分
        stem_cam = nn.Sequential(
        x = net_cam.conv1(x),       # (B, 64, 256, 256)   7×7 卷积，stride 2
        x = net_cam.bn1(x),         # (B, 64, 256, 256)
        x = net_cam.relu(x),        # (B, 64, 256, 256)
        x = net_cam.maxpool(x))      # (B, 64, 128, 128)   3×3 池化，stride 2

        stem_son = nn.Sequential(
        y = net_son.conv1(y),       # (B, 64, 256, 256)   7×7 卷积，stride 2
        y = net_son.bn1(y),         # (B, 64, 256, 256)
        y = net_son.relu(y),        # (B, 64, 256, 256)
        y = net_son.maxpool(y))      # (B, 64, 128, 128)   3×3 池化，stride 2

        c2_cam = net_cam.layer1(x)      # (B, 256,  128, 128)  分辨率 1/4
        c2_son = net_son.layer1(y)
        c3_cam = net_cam.layer2(c2_cam)     # (B, 512,  64,  64)   分辨率 1/8
        c3_son = net_son.layer2(c2_son)

        c3_son_self, c3_cam_self = self.self_attention(c3_cam,c3_son,0)

        self.c4 = net_cam.layer3(c3)     # (B, 1024, 32,  32)   分辨率 1/16
        self.c5 = net_cam.layer4(c4)     # (B, 2048, 16,  16)   分辨率 1/32


    def self_attention(self, cam_feats, sonar_feats, layer_num:int):
        #(channel_weight + spatial weight + 1) x feature map 
        cam_feats = (1 + self.camera_ca[layer_num](cam_feats) + self.camera_sa[layer_num](cam_feats)) * cam_feats
        sonar_feats = (1 + self.sonar_ca[layer_num](cam_feats) + self.sonar_sa[layer_num](cam_feats)) * sonar_feats
        return sonar_feats, cam_feats

    def cross_attention(self, cam_feats, sonar_feats, layer_num:int):
        
    
    def forward(self, cam, sonar):
        # 1. 两个 ResNet-50 分别提取特征
        cam_feats = self.features("camera", cam)
        sonar_feats = self.features("sonar", sonar)

        self_sonar_feats, self_cam_feats = self.self_attention(cam_feats, sonar_feats)

        feature_map = self.fpn(fused_feats)
        return feature_map


if __name__ == "__main__":
    model = DualResNet50().train()

    cam = torch.rand(2, 3, 512, 512)
    sonar = torch.rand(2, 3, 512, 512)

    with torch.no_grad():
        feature_map = model(cam, sonar)

    for name, feat in feature_map.items():
        print(name, tuple(feat.shape))
