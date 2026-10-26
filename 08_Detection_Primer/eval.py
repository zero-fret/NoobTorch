"""Average Precision for detection (single class, no training).

predictions: [(img_id, [x1, y1, x2, y2], score), ...]
groundtruths: [(img_id, [x1, y1, x2, y2]), ...]
"""
import numpy as np


def average_precision(predictions, groundtruths, iou_thresh=0.5):
    """Sort predictions by confidence; each is a TP if it matches a not-yet
    -matched GT in the same image above the IoU threshold, else FP.
    AP is the area under the precision-recall curve (VOC-style monotone
    interpolation), which is robust when recall jumps straight to 1."""
    if not predictions or not groundtruths:
        return 0.0

    preds = sorted(predictions, key=lambda p: p[2], reverse=True)
    from boxes import iou
    matched = {i: False for i in range(len(groundtruths))}
    tp, fp = [], []

    for img_id, box, _ in preds:
        best_iou, best_idx = 0.0, -1
        for i, (g_img, g_box) in enumerate(groundtruths):
            if g_img == img_id:
                v = iou(box, g_box)
                if v > best_iou:
                    best_iou, best_idx = v, i
        if best_iou >= iou_thresh and best_idx >= 0 and not matched[best_idx]:
            tp.append(1); fp.append(0); matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / len(groundtruths)
    return _ap_from_curve(precision, recall)


def _ap_from_curve(precision, recall):
    """VOC-style area under the PR curve: prepend a sentinel at recall=0,
    make precision non-increasing, then sum precision over recall steps."""
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([1.0], precision))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def map_50_95(predictions, groundtruths):
    """COCO-style mean over IoU thresholds 0.5..0.95."""
    thresholds = np.arange(0.5, 1.0, 0.05)
    return float(np.mean([average_precision(predictions, groundtruths, t)
                          for t in thresholds]))
