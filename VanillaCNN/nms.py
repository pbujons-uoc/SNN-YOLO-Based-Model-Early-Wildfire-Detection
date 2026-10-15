import torch

def box_iou(box1, box2):
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.size(0), box2.size(0)))

    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


def nms(dets, iou_thres=0.5):
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in dets])
    scores = torch.tensor([d[4] for d in dets])
    classes = torch.tensor([d[5] for d in dets])

    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(dets[i])

        if idxs.numel() == 1:
            break

        cur = boxes[i].unsqueeze(0)
        others = boxes[idxs[1:]]
        other_classes = classes[idxs[1:]]

        ious = box_iou(cur, others)[0]
        same_class = (classes[i] == other_classes)
        mask = (ious < iou_thres) | (~same_class)

        idxs = idxs[1:][mask]

    return keep
