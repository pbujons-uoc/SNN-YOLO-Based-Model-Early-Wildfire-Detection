import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleYoloLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, target):
        # preds/target: (B,S,S,5+C) with preds = [xy(2), wh(2), obj_logit(1), cls_logits(C)]
        pred_box = preds[..., 0:4]
        pred_obj_logit = preds[..., 4]
        pred_cls_logits = preds[..., 5:]

        tgt_box = target[..., 0:4]
        tgt_obj = target[..., 4]
        tgt_cls = target[..., 5:]

        obj_mask = tgt_obj > 0.5
        noobj_mask = ~obj_mask

        # Box regression (only when object exists)
        if obj_mask.any():
            box_loss = F.mse_loss(pred_box[obj_mask], tgt_box[obj_mask], reduction="sum")
        else:
            box_loss = torch.tensor(0.0, device=preds.device)

        # Objectness (all cells)
        obj_loss_all = self.bce_logits(pred_obj_logit, tgt_obj)
        obj_loss = obj_loss_all[obj_mask].sum() + self.lambda_noobj * obj_loss_all[noobj_mask].sum()

        # Classification (only when object exists)
        if obj_mask.any():
            cls_loss_all = self.bce_logits(pred_cls_logits[obj_mask], tgt_cls[obj_mask])
            cls_loss = cls_loss_all.sum()
        else:
            cls_loss = torch.tensor(0.0, device=preds.device)

        total = self.lambda_coord * box_loss + obj_loss + cls_loss
        bs = preds.size(0)

        # Return (total, box, "cls") where "cls" includes obj+cls for YOLO-like logging
        return total / bs, box_loss / bs, (obj_loss + cls_loss) / bs

