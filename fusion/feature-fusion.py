from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from util.CBAM import ChannelAttention, SpatialAttention

class DualResNet50(nn.Module):

    def __init__(self, model_camera, model_sonar, out_channel=512):
        super().__init__()
        self.in_channels_list = [256, 512, 1024, 2048]
        self.in_width_list = [128, 64, 32, 16]        
        self.camera_ca = nn.ModuleList([ChannelAttention(512),
        ChannelAttention(1024),
        ChannelAttention(2048)])
        self.camera_sa = nn.ModuleList([SpatialAttention(), SpatialAttention(), SpatialAttention()])
        #each layer need its own weight to extarct the feature

        self.sonar_ca = nn.ModuleList([ChannelAttention(512),
                ChannelAttention(1024),
                ChannelAttention(2048)])
        self.sonar_sa = nn.ModuleList([SpatialAttention(), SpatialAttention(), SpatialAttention()])

        self.son_mha_c3 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.son_mha_c4 = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.son_mha_c5 = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)

        self.cam_mha_c3 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.cam_mha_c4 = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.cam_mha_c5 = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)

        self.FPN_s = FeaturePyramidNetwork(in_channels_list=self.in_channels_list, out_channels=out_channel)
        self.FPN_c = FeaturePyramidNetwork(in_channels_list=self.in_channels_list, out_channels=out_channel)

        self.cam = model_camera
        self.sonar = model_sonar

    def features(self, x, y):
        """x is camera, y is sonar"""
        net_cam = self.cam
        net_son = self.sonar

        # stem：ResNet 的第一部分
        stem_cam = nn.Sequential(
        net_cam.conv1,       # (B, 64, 256, 256)   7×7 卷积，stride 2
        net_cam.bn1,         # (B, 64, 256, 256)
        net_cam.relu,        # (B, 64, 256, 256)
        net_cam.maxpool)      # (B, 64, 128, 128)   3×3 池化，stride 2

        stem_son = nn.Sequential(
        net_son.conv1,       # (B, 64, 256, 256)   7×7 卷积，stride 2
        net_son.bn1,         # (B, 64, 256, 256)
        net_son.relu,        # (B, 64, 256, 256)
        net_son.maxpool)      # (B, 64, 128, 128)   3×3 池化，stride 2

        x = stem_cam(x)
        y = stem_son(y)

        c2_cam = net_cam.layer1(x)      # (B, 256,  128, 128)  分辨率 1/4
        c2_son = net_son.layer1(y)
        c3_cam = net_cam.layer2(c2_cam)     # (B, 512,  64,  64)   分辨率 1/8
        c3_son = net_son.layer2(c2_son)

        c3_son_t = self.tokenize(c3_son)
        c3_cam_t = self.tokenize(c3_cam)
        c3_stoc = self.son_mha_c3(query=c3_cam_t, key=c3_son_t, value=c3_son_t, need_weights=False)
        c3_ctos = self.cam_mha_c3(query=c3_son_t, key=c3_cam_t, value=c3_cam_t, need_weights=False)
        c3_son_self, c3_cam_self = self.self_attention(c3_cam,c3_son,0)
        c3_son = c3_son_self + self.untokenize(c3_ctos, 1)
        c3_cam = c3_cam_self + self.untokenize(c3_stoc, 1)

        c4_son = net_son.layer3(c3_son)     # (B, 1024, 32,  32)   分辨率 1/16
        c4_cam = net_cam.layer3(c3_cam)

        c4_son_t = self.tokenize(c4_son)
        c4_cam_t = self.tokenize(c4_cam)
        c4_stoc = self.son_mha_c4(query=c4_cam_t, key=c4_son_t, value=c4_son_t, need_weights=False)
        c4_ctos = self.cam_mha_c4(query=c4_son_t, key=c4_cam_t, value=c4_cam_t, need_weights=False)
        c4_son_self, c4_cam_self = self.self_attention(c4_cam,c4_son,1)
        c4_son = c4_son_self + self.untokenize(c4_ctos, 2)
        c4_cam = c4_cam_self + self.untokenize(c4_stoc, 2)

        c5_son = net_son.layer4(c4_son)
        c5_cam = net_cam.layer4(c4_cam)
        c5_son_t = self.tokenize(c5_son)
        c5_cam_t = self.tokenize(c5_cam)
        c5_stoc = self.son_mha_c5(query=c5_cam_t, key=c5_son_t, value=c5_son_t, need_weights=False)
        c5_ctos = self.cam_mha_c5(query=c5_son_t, key=c5_cam_t, value=c5_cam_t, need_weights=False)
        c5_son_self, c5_cam_self = self.self_attention(c5_cam,c5_son,2)
        c5_son = c5_son_self + self.untokenize(c5_ctos,3)
        c5_cam = c5_cam_self + self.untokenize(c5_stoc,3)

        return {"c2":c2_son, "c3":c3_son, "c4":c4_son, "c5":c5_son}, {"c2":c2_cam, "c3":c3_cam, "c4":c4_cam, "c5":c5_cam}

    def self_attention(self, cam_feats, sonar_feats, layer_num:int):
        #(channel_weight + spatial weight + 1) x feature map 
        cam_feats = (1 + self.camera_ca[layer_num](cam_feats) + self.camera_sa[layer_num](cam_feats)) * cam_feats
        sonar_feats = (1 + self.sonar_ca[layer_num](sonar_feats) + self.sonar_sa[layer_num](sonar_feats)) * sonar_feats
        return sonar_feats, cam_feats

    def tokenize(self, feats):
        feats = feats.flatten(2).transpose(1,2) 
        # B,C,H,W ---> B,HW,C
        # q_proj = nn.Linear(self.in_channels_list[layer_num], dim)
        # v_proj = nn.Linear(self.in_channels_list[layer_num], dim)
        # k_proj = nn.Linear(self.in_channels_list[layer_num], dim)
        # q = q_proj(feats)
        # k = k_proj(feats)
        # v = v_proj(feats)
        return feats
    def untokenize(self, feats, layer_num):
        feats = feats.transpose(1,2)
        B = feats.shape[0]
        # B,HW,C ---> B, C, HW
        feats = feats.reshape(B, self.in_channels_list[layer_num], self.in_width_list[layer_num], self.in_width_list[layer_num])
        # B, C, HW --> B, C, H, W
        return feats


    
    def forward(self, cam, sonar):
        ''''input/output: camera, sonar'''
        sonar_feature, camera_feature = self.features(cam, sonar)
        cam_fpn = self.FPN_c(camera_feature)
        son_fpn = self.FPN_s(sonar_feature)       
        return cam_fpn, son_fpn
