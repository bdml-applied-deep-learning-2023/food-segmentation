import torch
from torch import nn
from torch.nn import functional as F
from torch.hub import load


class DINOv2Segmenter(nn.Module):

    def __init__(self, num_labels):
        super(DINOv2Segmenter, self).__init__()

        self.num_labels = num_labels
        self.feature_dim = 1024
        # self.feature_dim = 1536

        self.dinov2 = load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dinov2.requires_grad_(False)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.num_labels, 1),
        )

        self.exsistence_classifier = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.num_labels)
        )

    def forward(self, x, heatmap=None, existence=None):
        N, C, H, W = x.size()

        with torch.no_grad():
            outputs = self.dinov2.forward_features(x)
            patch_embedding = outputs["x_norm_patchtokens"]
            patch_embedding = patch_embedding.view(N, H//14, W//14, -1).permute(0, 3, 1, 2)

        logits_heatmap = self.classifier(patch_embedding)
        logits_heatmap = F.interpolate(logits_heatmap, size=(H, W))

        logits_existence = self.exsistence_classifier(patch_embedding)

        loss = None
        tp_acc = None
        tn_acc = None
        acc_heatmap = None
        acc_existence = None
        if heatmap is not None and existence is not None:
            # existence_mask = existence[:, :, None, None].expand(-1, -1, H, W)
            # loss_heatmap = F.binary_cross_entropy_with_logits(logits_heatmap[existence_mask == 1], heatmap[existence_mask == 1])
            loss_heatmap = F.binary_cross_entropy_with_logits(logits_heatmap, heatmap, reduction="none")
            loss_heatmap = loss_heatmap[heatmap >= 0.5].mean() + 3.0*loss_heatmap[heatmap < 0.5].mean()

            loss_existence = F.binary_cross_entropy_with_logits(logits_existence, existence, reduction="none")
            loss_existence = loss_existence[existence >= 0.5].mean() + 3.0*loss_existence[existence < 0.5].mean()
            pred_existence = logits_existence.sigmoid()

            loss = loss_heatmap + loss_existence
            
            pred = (logits_heatmap >= 0.5).float()
            target = (heatmap >= 0.5).float()

            tp_acc = (pred[target == 1] == 1).float().mean()
            tn_acc = (pred[target == 0] == 0).float().mean()
            acc_heatmap = 0.5*tp_acc + 0.5*tn_acc

            acc_existence = 0.5*(pred_existence[existence >= 0.5] >= 0.5).float().mean() + 0.5*(pred_existence[existence < 0.5] < 0.5).float().mean()
            # acc_existence = ((pred_existence >= 0.5).float() == (existence >= 0.5).float()).float().mean()

        return {
            "loss": loss,
            "tp_acc": tp_acc,
            "tn_acc": tn_acc,
            "acc_heatmap": acc_heatmap,
            "acc_existence": acc_existence,
            "logits_heatmap": logits_heatmap,
            "logits_existence": logits_existence,
        }


### checkpoints/dino_seg_bk
class DINOv2Segmenter2(nn.Module):

    def __init__(self, num_labels):
        super(DINOv2Segmenter2, self).__init__()

        self.num_labels = num_labels
        # self.feature_dim = 1024
        self.feature_dim = 1024

        self.dinov2 = load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        self.dinov2.requires_grad_(False)
        
        self.classifier = nn.Sequential(
            nn.ReflectionPad2d([1 for _ in range(4)]),
            nn.Conv2d(self.feature_dim, self.num_labels, 3),
        )

        self.exsistence_classifier = nn.Sequential(
            nn.ReflectionPad2d([1 for _ in range(4)]),
            nn.Conv2d(self.feature_dim, self.feature_dim, 3),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.num_labels)
        )

    def forward(self, x, heatmap=None, existence=None):
        N, C, H, W = x.size()

        with torch.no_grad():
            outputs = self.dinov2.backbone.forward_features(x)
            patch_embedding = outputs["x_norm_patchtokens"]
            patch_embedding = patch_embedding.view(N, H//14, W//14, -1).permute(0, 3, 1, 2)

        logits_heatmap = self.classifier(patch_embedding)
        logits_heatmap = F.interpolate(logits_heatmap, size=(H, W))

        logits_existence = self.exsistence_classifier(patch_embedding)

        loss = None
        tp_acc = None
        tn_acc = None
        acc_heatmap = None
        acc_existence = None
        if heatmap is not None and existence is not None:
            # existence_mask = existence[:, :, None, None].expand(-1, -1, H, W)
            # loss_heatmap = F.binary_cross_entropy_with_logits(logits_heatmap[existence_mask == 1], heatmap[existence_mask == 1])
            loss_heatmap = F.binary_cross_entropy_with_logits(logits_heatmap, heatmap, reduction="none")
            loss_heatmap = loss_heatmap[heatmap >= 0.5].mean() + 3.0*loss_heatmap[heatmap < 0.5].mean()

            loss_existence = F.binary_cross_entropy_with_logits(logits_existence, existence, reduction="none")
            loss_existence = loss_existence[existence >= 0.5].mean() + 3.0*loss_existence[existence < 0.5].mean()
            pred_existence = logits_existence.sigmoid()

            loss = loss_heatmap + loss_existence
            
            pred = (logits_heatmap >= 0.5).float()
            target = (heatmap >= 0.5).float()

            tp_acc = (pred[target == 1] == 1).float().mean()
            tn_acc = (pred[target == 0] == 0).float().mean()
            acc_heatmap = 0.5*tp_acc + 0.5*tn_acc

            acc_existence = 0.5*(pred_existence[existence >= 0.5] >= 0.5).float().mean() + 0.5*(pred_existence[existence < 0.5] < 0.5).float().mean()
            # acc_existence = ((pred_existence >= 0.5).float() == (existence >= 0.5).float()).float().mean()

        return {
            "loss": loss,
            "tp_acc": tp_acc,
            "tn_acc": tn_acc,
            "acc_heatmap": acc_heatmap,
            "acc_existence": acc_existence,
            "logits_heatmap": logits_heatmap,
            "logits_existence": logits_existence,
        }

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
