import torch
import torch.nn as nn
import torch.nn.functional as F

class Semantic_Enhancement_Module(nn.Module):
    def __init__(self, in_dim=512, num_class=15, groups=16):
        super().__init__()
        self.groups = groups
        self.num_class = num_class
        
        # 标签下采样与通道对齐
        self.label_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),       # 将224x224下采样到16x16
            nn.Conv2d(num_class, in_dim, 3, padding=1),  # 通道数15->512
            nn.GroupNorm(16, in_dim),
            nn.ReLU()
        )
        
        # 动态特征增强
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim//2, 1),   # 融合标签和视觉特征
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        """
        输入:
        x:     (B, N, C) 例如 [1, 256, 512]
        label: (B, num_class, H, W) 例如 [1,15,224,224]
        """
        B, N, C = x.shape
        H = W = int(N**0.5)
       
        if label is None:
            label = torch.zeros((B, self.num_class, 224, 224)).to(x.device)
        
        # 转换视觉特征到4D
        x_4d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, 16, 16)
        
        # --------------------------------------
        # 分阶段处理标签特征
        label_feat = F.interpolate(label, size=(64, 64), mode='bilinear')  # 中间下采样
        label_feat = self.label_processor(label_feat)  # (B, 512, 16, 16)

        # # 处理多通道标签
        # label_feat = self.label_processor(label)  # (B, 512, 16, 16)
        
        # 通道注意力
        channel_weight = self.channel_att(
            torch.cat([x_4d, label_feat], dim=1)  # (B, 1024, 16, 16)
        )  # (B, 512, 16, 16)
        
        # 空间注意力
        spatial_weight = self.spatial_att(x_4d)  # (B, 1, 16, 16)

        # 稳定特征融合
        enhanced_feat = x_4d * (0.6 + channel_weight) * (0.4 + spatial_weight)
        
        # # 特征融合
        # enhanced_feat = x_4d * channel_weight * spatial_weight
        enhanced_feat = enhanced_feat.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # # 恢复原始维度
        # return enhanced_feat.permute(0, 2, 3, 1).view(B, N, C)
        return enhanced_feat
        # ---------------------------------------

        # # ---------------without channelatt------------
        # label_feat = F.interpolate(label, size=(64, 64), mode='bilinear')  # 中间下采样
        # label_feat = self.label_processor(label_feat)  # (B, 512, 16, 16)
        
        # # 空间注意力
        # spatial_weight = self.spatial_att(x_4d)  # (B, 1, 16, 16)

        # # 稳定特征融合
        # # enhanced_feat = x_4d * (0.5 + channel_weight) * (0.5 + spatial_weight)
        # enhanced_feat = x_4d  * (1 + spatial_weight)

        # # # 特征融合
        # # enhanced_feat = x_4d * channel_weight * spatial_weight
        # enhanced_feat = enhanced_feat.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # # # 恢复原始维度
        # # return enhanced_feat.permute(0, 2, 3, 1).view(B, N, C)
        # return enhanced_feat
        # # ------------------------------------------
    
        # # -------------without spatialatt--------------------
        #         # 分阶段处理标签特征
        # label_feat = F.interpolate(label, size=(64, 64), mode='bilinear')  # 中间下采样
        # label_feat = self.label_processor(label_feat)  # (B, 512, 16, 16)

        # # # 处理多通道标签
        # # label_feat = self.label_processor(label)  # (B, 512, 16, 16)
        
        # # 通道注意力
        # channel_weight = self.channel_att(
        #     torch.cat([x_4d, label_feat], dim=1)  # (B, 1024, 16, 16)
        # )  # (B, 512, 16, 16)
        
        # # 空间注意力
        # # spatial_weight = self.spatial_att(x_4d)  # (B, 1, 16, 16)

        # # 稳定特征融合
        # # enhanced_feat = x_4d * (0.5 + channel_weight) * (0.5 + spatial_weight)
        # enhanced_feat = x_4d * (1 + channel_weight)
        
        # # # 特征融合
        # # enhanced_feat = x_4d * channel_weight * spatial_weight
        # enhanced_feat = enhanced_feat.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # # # 恢复原始维度
        # # return enhanced_feat.permute(0, 2, 3, 1).view(B, N, C)
        # return enhanced_feat
        # # -----------------------------------------------------




# import torch.nn as nn
# import torch
# from torch.nn.parameter import Parameter
# from torch.nn import functional as F

# class Semantic_Enhancement_Module(nn.Module):
#     def __init__(self, in_channels=512, num_class=15, groups=16):  # 修改参数顺序和默认值
#         super().__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # 调整线性层维度匹配
#         self.label_w = nn.Sequential(
#             nn.Linear(num_class, in_channels),
#             nn.ReLU(),
#             nn.Linear(in_channels, in_channels)
#         )
        
#         # 增加通道对齐卷积
#         self.channel_align = nn.Conv2d(in_channels, in_channels, 1)
        
#         # 参数初始化调整
#         self.weight = Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias = Parameter(torch.ones(1, groups, 1, 1))
#         self.sig = nn.Sigmoid()
        
#         # 初始化参数
#         nn.init.normal_(self.label_w[0].weight, std=0.02)
#         nn.init.normal_(self.label_w[2].weight, std=0.02)

#     def forward(self, x, label):
#         # 输入形状处理
#         b, c, h, w = x.size()  # 确保输入已经是4D形状
        
#         # 标签处理（假设label是one-hot编码）
#         label_emb = self.label_w(label)  # (b, c)
#         label_emb = label_emb.view(b, c, 1, 1).expand_as(x)  # (b, c, h, w)
        
#         # 通道对齐
#         x = self.channel_align(x)
        
#         # 特征增强
#         x_in = x + label_emb
#         xn = x * self.avg_pool(x_in)
#         xn = xn.sum(dim=1, keepdim=True)  # (b, 1, h, w)
        
#         # 分组标准化
#         t = xn.view(b * self.groups, -1)
#         t = t - t.mean(dim=1, keepdim=True)
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std
#         t = t.view(b, self.groups, h, w)
#         t = t * self.weight + self.bias
#         t = t.view(b * self.groups, 1, h, w)
        
#         # 门控融合
#         x = x * self.sig(t)
#         return x.view(b, c, h, w)