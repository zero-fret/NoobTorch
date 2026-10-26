"""Runnable tour of every detection concept, on made-up boxes.
No network, no dataset -- understand the output format first.

Run:  python demo.py
"""
from boxes import iou, normalized_to_pixels, grid_cell_of, cell_offset
from nms import nms
from eval import average_precision, map_50_95

S = 20          # 20x20 grid, like the Simple YOLO lessons
IMG = 640

# --- 1. Ground truth: one circle in image 0 (YOLO label) -------------------
gt_label = (0.30, 0.30, 40 / IMG, 40 / IMG)   # cx, cy, w, h, normalized
gt_box = normalized_to_pixels(gt_label, IMG)
print('GT box (pixels):', [round(v, 1) for v in gt_box])

# --- 2. A network's raw predictions for the same object --------------------
# A good box, a near-duplicate of it (same object, neighboring cell), and
# a low-confidence false positive.
good = [gt_box[0] + 2, gt_box[1] + 2, gt_box[2] + 2, gt_box[3] + 2, 0.94]
dup = [gt_box[0] + 8, gt_box[1] + 6, gt_box[2] + 8, gt_box[3] + 6, 0.78]
fp = [500, 500, 540, 540, 0.25]
raw = [good, dup, fp]

print('\nIoU good vs GT:', round(iou(good[:4], gt_box), 3))
print('IoU duplicate vs GT:', round(iou(dup[:4], gt_box), 3))

# --- 3. Confidence threshold + NMS -----------------------------------------
CONF = 0.5
before_nms = [b for b in raw if b[4] > CONF]
kept = nms(before_nms, iou_thresh=0.5)
print(f'\nRaw boxes above {CONF}: {len(before_nms)} -> after NMS: {len(kept)}')
print('Kept box score:', kept[0][4])

# --- 4. Average precision ---------------------------------------------------
preds = [(0, b[:4], b[4]) for b in raw]
gts = [(0, gt_box)]
print('\nAP@0.5:', round(average_precision(preds, gts, 0.5), 3))
print('mAP@0.5:0.95:', round(map_50_95(preds, gts), 3),
      '(low: the false positive and weak duplicate drag it down)')

# --- 5. Where the box would live in the SxS grid ---------------------------
gx, gy = grid_cell_of(gt_label[0], gt_label[1], S)
ox, oy = cell_offset(gt_label[0], gt_label[1], S)
print(f'\nGrid cell owning the center: ({gx},{gy}), inside-cell offset ({ox:.2f},{oy:.2f})')
print('=> that is exactly the (tx, ty) a YOLO head learns in lesson 09.')

# --- 6. Optional picture ----------------------------------------------------
try:
    import cv2
    import numpy as np
    bg = cv2.imread('../assets/backgrounds/000000001685.jpg')
    canvas = cv2.resize(bg, (IMG, IMG)) if bg is not None else np.full((IMG, IMG, 3), 30, dtype='uint8')
    for x in range(0, IMG, IMG // S):
        cv2.line(canvas, (x, 0), (x, IMG), (40, 40, 40), 1)
        cv2.line(canvas, (0, x), (IMG, x), (40, 40, 40), 1)
    def draw(box, color, label):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    draw(gt_box, (0, 255, 0), 'GT')
    draw(kept[0][:4], (0, 255, 255), f"pred {kept[0][4]:.2f}")
    cv2.imwrite('primer_demo.png', canvas)
    print('\nSaved primer_demo.png (green=GT, yellow=prediction, faint grid=SxS).')
except Exception as e:
    print(f'(Skipped drawing demo image: {e})')
