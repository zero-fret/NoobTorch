import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import CircleDataset, S, IMG_SIZE, decode_predictions, nms, compute_ap, decode_gt
from model import SimpleYOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH, IOU_THRESH = 0.5, 0.5

model = SimpleYOLO(S=S).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=False))
model.eval()

loader = DataLoader(CircleDataset('val'), batch_size=1, shuffle=False)
os.makedirs('test_result', exist_ok=True)

preds, gts = [], []
with torch.no_grad():
    for img_id, (img, target, mask) in enumerate(loader):
        boxes = decode_predictions(model(img.to(device)), S, CONF_THRESH, IMG_SIZE)
        boxes = nms(boxes, IOU_THRESH)
        for b in boxes:
            preds.append((img_id, b[:4], b[4], 0))
        gts.extend(decode_gt(target[0], mask[0], img_id, S))

        canvas = cv2.cvtColor((img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                              cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2, conf in boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f'{conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f'test_result/image_{img_id:04d}.jpg', canvas)

print(f'AP@0.5: {100*compute_ap(preds, gts, 0.5):.2f}%')
print(f'mAP@0.5:0.95: {100*np.mean([compute_ap(preds, gts, t) for t in np.arange(0.5,1.0,0.05)]):.2f}%')
