import torch
import torch.nn as nn
from torchvision import models


class HazardAwareAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch = self.channel_att(x) * x

        avg_pool = torch.mean(ch, dim=1, keepdim=True)
        max_pool = torch.max(ch, dim=1, keepdim=True)[0]

        sp = torch.cat([avg_pool, max_pool], dim=1)
        sp = self.spatial_att(sp) * ch

        return sp


class MHViTModel(nn.Module):
    """Legacy name wrapper – actual model implementation.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        self.vit = models.vit_b_16(weights=None)

        embed_dim = self.vit.hidden_dim

        self.multi_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(3)
        ])

        self.att = HazardAwareAttention(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.vit._process_input(x)

        cls_tok = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tok, x], dim=1)

        x = self.vit.encoder(x)

        patches = x[:, 1:, :]
        z = patches.mean(dim=1)

        for proj in self.multi_proj:
            z = z + proj(z)

        B, N, C = patches.shape
        H = W = int(N ** 0.5)

        att_map = self.att(
            patches.permute(0, 2, 1).view(B, C, H, W)
        )

        att_vec = att_map.mean(dim=[2, 3])

        out = self.classifier(z + att_vec)
        return out  # <--- Yeh missing tha, add kar diya gaya hai


# Alias for backward compatibility
MHViT = MHViTModel
