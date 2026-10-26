import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import CircleDataset, S, IMG_SIZE, decode_predictions, nms, compute_ap, decode_gt
from model import SimpleYOLO_DSConv_Coord

# Set this to match the weights you want to evaluate:
#   False -> control (model_coords_False.pth), True -> treatment (model_coords_True.pth)
USE_COORDS = False
WEIGHTS = f"model_coords_{USE_COORDS}.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH, IOU_THRESH = 0.3, 0.5

model = SimpleYOLO_DSConv_Coord(S=S, use_coords=USE_COORDS).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=False))
model.eval()

loader = DataLoader(CircleDataset('val'), batch_size=1, shuffle=False)
preds, gts = [], []
with torch.no_grad():
    for img_id, (img, target, mask) in enumerate(loader):
        boxes = nms(decode_predictions(model(img.to(device)), S, CONF_THRESH, IMG_SIZE), IOU_THRESH)
        for b in boxes:
            preds.append((img_id, b[:4], b[4], 0))
        gts.extend(decode_gt(target[0], mask[0], img_id, S))

ap50 = compute_ap(preds, gts, 0.5)
ap5095 = np.mean([compute_ap(preds, gts, t) for t in np.arange(0.5, 1.0, 0.05)])
print(f'coords={USE_COORDS}  AP@0.5: {100*ap50:.2f}%  mAP@0.5:0.95: {100*ap5095:.2f}%')
