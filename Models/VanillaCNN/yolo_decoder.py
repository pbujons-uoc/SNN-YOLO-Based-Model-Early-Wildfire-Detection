import torch

def decode_predictions(preds, conf_thres=0.25, S=20, img_size=640):
    """
    Decode grid predictions into a list of detections per image.

    Assumes preds is (B, S, S, 5+C) with:
      - preds[..., 0:2] = (cx, cy) offsets within cell in [0,1]
      - preds[..., 2:4] = (w, h) relative to full image in [0,1]
      - preds[..., 4]   = objectness LOGIT (raw)
      - preds[..., 5:]  = class LOGITS (raw)

    Returns:
      results: List[List[Tuple[x1,y1,x2,y2,score,cls]]]
               coords are in pixel units
    """
    B, S1, S2, D = preds.shape
    assert S1 == S and S2 == S, f"Expected grid {S}x{S}, got {S1}x{S2}"

    results = []
    for b in range(B):
        p = preds[b]  # (S, S, 5+C)
        dets = []

        # Objectness + class probabilities
        obj = torch.sigmoid(p[..., 4])         # (S, S)
        cls_scores = torch.sigmoid(p[..., 5:]) # (S, S, C)

        best_cls_score, best_cls = cls_scores.max(dim=-1)  # (S, S)
        conf = obj * best_cls_score                         # (S, S)

        ys, xs = torch.where(conf > conf_thres)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx, cy, w, h = p[y, x, 0:4]

            score = float(conf[y, x].item())
            cls = int(best_cls[y, x].item())

            # center in normalized image coordinates
            cx = (x + float(cx.item())) / S
            cy = (y + float(cy.item())) / S

            # w,h are already relative to full image (0..1): DO NOT divide by S
            w = float(w.item())
            h = float(h.item())

            # convert to pixel corners
            x1 = (cx - w / 2.0) * img_size
            y1 = (cy - h / 2.0) * img_size
            x2 = (cx + w / 2.0) * img_size
            y2 = (cy + h / 2.0) * img_size

            dets.append((x1, y1, x2, y2, score, cls))

        results.append(dets)

    return results
