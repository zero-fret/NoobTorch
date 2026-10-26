import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

S = 20
IMG_SIZE = 640


def _ap_from_curve(precision, recall):
    """VOC-style area under the PR curve: sentinel at recall=0, monotone
    non-increasing precision, then sum over recall steps."""
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([1.0], precision))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


class CircleDataset(Dataset):
    """Reads ALL circles in each label (this dataset has 4 circles per image)."""
    def __init__(self, split='train'):
        self.img_dir = f'data/{split}/images'
        self.label_dir = f'data/{split}/labels'
        self.image_files = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = torch.zeros((5, S, S), dtype=torch.float32)
        mask = torch.zeros((S, S), dtype=torch.bool)
        with open(os.path.join(self.label_dir, name.replace('.png', '.txt'))) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                _, xc, yc, w, h = line.split()
                xc, yc, w, h = map(float, (xc, yc, w, h))
                gx = min(max(int(xc * S), 0), S - 1)
                gy = min(max(int(yc * S), 0), S - 1)
                # Offset from the cell's top-left corner, in [0, 1)
                ox = min(max(xc * S - gx, 0.0), 1.0 - 1e-6)
                oy = min(max(yc * S - gy, 0.0), 1.0 - 1e-6)
                target[0, gy, gx] = ox
                target[1, gy, gx] = oy
                target[2, gy, gx] = w
                target[3, gy, gx] = h
                target[4, gy, gx] = 1.0
                mask[gy, gx] = True
        return img, target, mask


def decode_predictions(output, S, conf_thresh, img_size=IMG_SIZE):
    output = output.squeeze(0)
    tx = torch.sigmoid(output[0])
    ty = torch.sigmoid(output[1])
    tw = output[2]                # width/height regressed directly (no sigmoid),
    th = output[3]                # like YOLO-style raw size prediction
    conf = torch.sigmoid(output[4])

    boxes = []
    for i in range(S):
        for j in range(S):
            if conf[i, j] > conf_thresh:
                cx = (j + tx[i, j]) / S * img_size
                cy = (i + ty[i, j]) / S * img_size
                w = max(0.0, tw[i, j] * img_size)
                h = max(0.0, th[i, j] * img_size)
                boxes.append([float(cx - w / 2), float(cy - h / 2),
                              float(cx + w / 2), float(cy + h / 2), float(conf[i, j])])
    return boxes


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-6)


def nms(boxes, iou_thresh):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if box_iou(best, b) < iou_thresh]
    return keep


def compute_ap(predictions, groundtruths, iou_thresh=0.5):
    preds = sorted((p for p in predictions if p[3] == 0), key=lambda p: p[2], reverse=True)
    gts = [g for g in groundtruths if g[2] == 0]
    if not preds or not gts:
        return 0.0
    matched = [False] * len(gts)
    tp, fp = [], []
    for img_id, box, conf, _ in preds:
        best_iou, best_idx = 0.0, -1
        for i, (gid, gbox, _) in enumerate(gts):
            if gid == img_id:
                v = box_iou(box, gbox)
                if v > best_iou:
                    best_iou, best_idx = v, i
        if best_iou >= iou_thresh and best_idx >= 0 and not matched[best_idx]:
            tp.append(1); fp.append(0); matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / len(gts)
    return _ap_from_curve(precision, recall)


def compute_map(predictions, groundtruths, iou_thresh=0.5):
    return compute_ap(predictions, groundtruths, iou_thresh)


def decode_gt(targets, masks, img_id, S, img_size=IMG_SIZE):
    """GT boxes for every positive cell (left-edge offset convention)."""
    out = []
    gy, gx = np.where(masks.cpu().numpy())
    for y, x in zip(gy, gx):
        tx, ty = targets[0, y, x].item(), targets[1, y, x].item()
        w, h = targets[2, y, x].item(), targets[3, y, x].item()
        cx, cy = (x + tx) / S * img_size, (y + ty) / S * img_size
        w, h = w * img_size, h * img_size
        out.append((img_id, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0))
    return out
