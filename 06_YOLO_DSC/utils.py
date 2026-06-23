import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

S = 20
IMG_SIZE = 640

class CircleDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.img_dir = f'data/{split}/images'
        self.label_dir = f'data/{split}/labels'
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # (C, H, W)

        target = torch.zeros((5, S, S), dtype=torch.float32)
        mask = torch.zeros((S, S), dtype=torch.bool)
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                grid_x = int(x_center * S)
                grid_y = int(y_center * S)
                grid_x = min(grid_x, S - 1)
                grid_y = min(grid_y, S - 1)

                offset_x = x_center * S - grid_x
                offset_y = y_center * S - grid_y

                target[0, grid_y, grid_x] = offset_x
                target[1, grid_y, grid_x] = offset_y
                target[2, grid_y, grid_x] = width
                target[3, grid_y, grid_x] = height
                target[4, grid_y, grid_x] = 1.0
                mask[grid_y, grid_x] = True

        return img, target, mask


def decode_predictions(output, S, conf_thresh, img_size=640):
    output = output.squeeze(0)          # (5, S, S)
    tx = torch.sigmoid(output[0])       # (S, S)
    ty = torch.sigmoid(output[1])
    tw = torch.sigmoid(output[2])
    th = torch.sigmoid(output[3])
    conf = torch.sigmoid(output[4])

    boxes = []
    for i in range(S):
        for j in range(S):
            if conf[i, j] > conf_thresh:
                cx = (j + tx[i, j]) / S * img_size
                cy = (i + ty[i, j]) / S * img_size
                w = tw[i, j] * img_size
                h = th[i, j] * img_size
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append([float(x1), float(y1), float(x2), float(y2), float(conf[i, j])])
    return boxes


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def nms(boxes, iou_thresh):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [box for box in boxes if box_iou(best, box) < iou_thresh]
    return keep

def compute_ap(predictions, groundtruths, iou_thresh=0.5):

    class_preds = [p for p in predictions if p[3] == 0]
    class_gts = [g for g in groundtruths if g[2] == 0]
    
    if not class_gts or not class_preds:
        return 0.0

    class_preds = sorted(class_preds, key=lambda x: x[2], reverse=True)
    gt_matched = {i: False for i in range(len(class_gts))}
    tp, fp = [], []
    
    for pred in class_preds:
        img_id = pred[0]
        img_gts = [(idx, gt) for idx, gt in enumerate(class_gts) if gt[0] == img_id]
        if not img_gts:
            fp.append(1)
            tp.append(0)
            continue
        
        best_iou = 0.0
        best_global_idx = -1
        for global_idx, gt in img_gts:
            iou = box_iou(pred[1], gt[1])
            if iou > best_iou:
                best_iou = iou
                best_global_idx = global_idx
        
        if best_iou >= iou_thresh and not gt_matched[best_global_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[best_global_idx] = True
        else:
            fp.append(1)
            tp.append(0)
    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / len(class_gts)

    if len(precisions) < 2:
        return precisions[0] if len(precisions) == 1 else 0.0
    
    ap = np.trapezoid(precisions, recalls)
    return ap


def compute_map(predictions, groundtruths, iou_thresh=0.5, num_classes=1):
    return compute_ap(predictions, groundtruths, iou_thresh)