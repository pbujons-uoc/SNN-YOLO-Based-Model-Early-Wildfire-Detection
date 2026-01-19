import torch
from pathlib import Path
from nms import box_iou, nms
from yolo_decoder import decode_predictions


# -------------------------------------------------------------
# Compute AP
# -------------------------------------------------------------
def compute_ap(recall, precision):
    """
    Standard PR-curve integration:
    AP = area under precision-recall curve
    """
    if recall.numel() == 0:
        return 0.0

    # Add boundary points
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])

    # Make precision monotonically decreasing
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    # Integrate where recall changes
    idx = (mrec[1:] != mrec[:-1]).nonzero().flatten()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


# -------------------------------------------------------------
# Load ground truth boxes from YOLO labels
# -------------------------------------------------------------
def load_gt_from_yolo(labels_dir, img_files, img_size=640):
    """
    Loads GT boxes from YOLO txt labels.
    Returns:
        dict: image_name -> list of (x1,y1,x2,y2,cls)
    """
    labels_dir = Path(labels_dir)
    gt_dict = {}

    for img_name in img_files:
        stem = Path(img_name).stem
        label_path = labels_dir / f"{stem}.txt"
        boxes = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = map(float, parts)
                    cls = int(cls)

                    # Convert normalized xywh → absolute x1y1x2y2
                    cx_abs = cx * img_size
                    cy_abs = cy * img_size
                    w_abs = w * img_size
                    h_abs = h * img_size

                    x1 = cx_abs - w_abs / 2
                    y1 = cy_abs - h_abs / 2
                    x2 = cx_abs + w_abs / 2
                    y2 = cy_abs + h_abs / 2

                    boxes.append((x1, y1, x2, y2, cls))

        gt_dict[img_name] = boxes

    return gt_dict


# -------------------------------------------------------------
# Internal: compute mAP at a single IoU threshold using precomputed preds/gt
# -------------------------------------------------------------
def _map_at_iou(all_predictions, gt_dict, class_names, iou_thres):
    aps = []
    per_class_results = {}

    for c, cname in enumerate(class_names):
        # Filter predictions of this class
        preds_c = [p for p in all_predictions if p[6] == c]
        preds_c.sort(key=lambda x: x[5], reverse=True)  # sort by score

        # Count GT instances
        npos = 0
        gt_boxes_c = {}

        for img_name, gts in gt_dict.items():
            gts_c = [g for g in gts if g[4] == c]
            boxes = (
                torch.tensor([[g[0], g[1], g[2], g[3]]], dtype=torch.float32).repeat(0, 1)
                if False else None
            )
            if gts_c:
                boxes = torch.tensor([[g[0], g[1], g[2], g[3]] for g in gts_c], dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)

            gt_boxes_c[img_name] = {"boxes": boxes, "used": [False] * len(gts_c)}
            npos += len(gts_c)

        # No GT → AP = 0
        if npos == 0:
            per_class_results[cname] = {"AP": 0.0, "P": 0.0, "R": 0.0}
            aps.append(0.0)
            continue

        tp = torch.zeros(len(preds_c), dtype=torch.float32)
        fp = torch.zeros(len(preds_c), dtype=torch.float32)

        # Match predictions to GT
        for i, pred in enumerate(preds_c):
            img_name, x1, y1, x2, y2, score, cls = pred
            gt_info = gt_boxes_c[img_name]
            gt_boxes = gt_info["boxes"]
            used = gt_info["used"]

            if gt_boxes.numel() == 0:
                fp[i] = 1.0
                continue

            pred_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            ious = box_iou(pred_box, gt_boxes)[0]
            best_iou, best_idx = ious.max(dim=0)

            best_idx_int = int(best_idx.item())
            if float(best_iou.item()) >= float(iou_thres) and not used[best_idx_int]:
                tp[i] = 1.0
                used[best_idx_int] = True
                gt_boxes_c[img_name]["used"] = used
            else:
                fp[i] = 1.0

        # Precision–Recall
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recall = cum_tp / (npos + 1e-6)
        precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

        ap = compute_ap(recall, precision)
        aps.append(ap)

        # last point of curve
        P_final = float(precision[-1].item()) if precision.numel() else 0.0
        R_final = float(recall[-1].item()) if recall.numel() else 0.0

        per_class_results[cname] = {"AP": ap, "P": P_final, "R": R_final}

    mAP = sum(aps) / max(len(aps), 1)
    return float(mAP), per_class_results


# -------------------------------------------------------------
# mAP Evaluation (supports mAP@0.5 and mAP@0.5:0.95)
# -------------------------------------------------------------
def evaluate_map(
    model,
    val_ds,
    val_loader,
    labels_dir,
    class_names,
    S=20,
    img_size=640,
    iou_thres=0.5,
    conf_thres_eval=0.001,
    nms_iou_thres=0.5,
):
    """
    If iou_thres is a float:
        returns: mAP50, per_class_results

    If iou_thres is a list/tuple/tensor of IoU thresholds (e.g. 0.50..0.95 step 0.05):
        returns: mAP50, mAP50_95, per_class_results   (per_class_results is for IoU=0.5)
    """
    model.eval()
    img_files = val_ds.img_files
    gt_dict = load_gt_from_yolo(labels_dir, img_files, img_size=img_size)

    # Collect predictions once
    all_predictions = []  # (img_name, x1,y1,x2,y2,score,cls)

    device = next(model.parameters()).device
    with torch.no_grad():
        idx = 0
        for images, _ in val_loader:
            images = images.to(device)
            preds = model(images)

            batch_dets = decode_predictions(
                preds,
                conf_thres=conf_thres_eval,
                S=S,
                img_size=img_size
            )

            for dets in batch_dets:
                img_name = img_files[idx]
                dets = nms(dets, iou_thres=nms_iou_thres)  # use provided NMS threshold
                for (x1, y1, x2, y2, score, cls) in dets:
                    all_predictions.append((img_name, x1, y1, x2, y2, score, cls))
                idx += 1

    # Determine whether we compute single-IoU or COCO-style
    if isinstance(iou_thres, (list, tuple)):
        iou_list = [float(x) for x in iou_thres]
    elif torch.is_tensor(iou_thres) and iou_thres.numel() > 1:
        iou_list = [float(x) for x in iou_thres.flatten().tolist()]
    else:
        iou_list = None

    if iou_list is None:
        # Backward-compatible path: only one threshold
        mAP50, per_class_results = _map_at_iou(all_predictions, gt_dict, class_names, float(iou_thres))
        return mAP50, per_class_results

    # COCO-style: compute mAP for each IoU threshold and average
    maps = []
    per_class_results_50 = None
    for t in iou_list:
        m, per_cls = _map_at_iou(all_predictions, gt_dict, class_names, float(t))
        maps.append(m)
        if abs(t - 0.5) < 1e-9:
            per_class_results_50 = per_cls

    mAP50 = maps[iou_list.index(0.5)] if 0.5 in iou_list else maps[0]
    mAP50_95 = float(sum(maps) / max(len(maps), 1))

    if per_class_results_50 is None:
        # if user didn't include 0.5 explicitly, fall back to first threshold per-class
        _, per_class_results_50 = _map_at_iou(all_predictions, gt_dict, class_names, float(iou_list[0]))

    return mAP50, mAP50_95, per_class_results_50
