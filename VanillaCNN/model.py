import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleYoloCNN(nn.Module):
    """
    Very vanilla CNN that outputs one YOLO-style prediction per grid cell.
    Output shape: (B, S, S, 5 + num_classes)
    """
    def __init__(self, num_classes=2, img_size=640, S=20):
        super().__init__()
        self.num_classes = num_classes
        self.S = S

        # Use Pre-trained ResNet18 Backbone
        # We need torchvision (imported inside or at top)
        from torchvision import models
        
        # Load pre-trained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # ResNet18 structure:
        # (conv1, bn1, relu, maxpool) -> layer1 -> layer2 -> layer3 -> layer4 -> (avgpool, fc)
        # We want the output of layer4.
        
        # Taking all children except the last 2 (avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # ResNet18 layer4 outputs 512 channels.
        # With input 640x640, the output stride is 32, so spatial size is 20x20.
        # This matches our S=20 requirement perfectly.

        # Head: 512 ch -> (5 + C)
        self.head = nn.Conv2d(512, 5 + num_classes, 1)

    def forward(self, x):
        feat = self.features(x)          # (B, 512, 20, 20)
        out = self.head(feat)            # (B, 5 + C, 20, 20)
        out = out.permute(0, 2, 3, 1)    # (B, S, S, 5 + C)

        # YOLO-style parameterization
        xy = torch.sigmoid(out[..., 0:2])      # cx, cy within cell
        wh = torch.sigmoid(out[..., 2:4])      # w, h relative to image
        obj_logits = out[..., 4:5]             # raw logits
        cls_logits = out[..., 5:]              # raw logits

        return torch.cat([xy, wh, obj_logits, cls_logits], dim=-1)
