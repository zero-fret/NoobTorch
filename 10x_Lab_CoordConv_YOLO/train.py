import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import CircleDataset, S, decode_predictions, nms, compute_ap, decode_gt
from model import SimpleYOLO_DSConv_Coord

BATCH = 32
EPOCHS = 60
LR = 0.001
COORD_W = 10.0
# Toggle this line, then rerun once per arm:
#   USE_COORDS=False -> control, USE_COORDS=True -> treatment.
# Weights save to model_coords_False.pth / model_coords_True.pth.
USE_COORDS = False


def forward_loss(model, mse, imgs, targets, masks):
    out = model(imgs)
    pred_tx = torch.sigmoid(out[:, 0:1])
    pred_ty = torch.sigmoid(out[:, 1:2])
    pred_tw, pred_th = out[:, 2:3], out[:, 3:4]
    coord = mse(pred_tx, targets[:, 0:1]) + mse(pred_ty, targets[:, 1:2]) \
          + mse(pred_tw, targets[:, 2:3]) + mse(pred_th, targets[:, 3:4])
    mask = masks.unsqueeze(1)
    coord = (coord * mask.float()).sum() / (mask.sum() + 1e-6) * COORD_W
    num_pos = mask.sum().clamp(min=1)
    num_neg = float(S * S * imgs.size(0)) - num_pos
    pos_weight = torch.tensor([(num_neg / num_pos).item()], device=imgs.device)
    conf_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out[:, 4:5], targets[:, 4:5])
    return coord + conf_loss


@torch.no_grad()
def validate(model, use_coords, val_loader, device):
    model.eval()
    preds, gts = [], []
    for b, (imgs, targets, masks) in enumerate(val_loader):
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            img_id = b * imgs.size(0) + i
            for box in nms(decode_predictions(model(imgs[i:i+1]), S, 0.1), 0.5):
                preds.append((img_id, box[:4], box[4], 0))
            gts.extend(decode_gt(targets[i], masks[i], img_id, S))
    if preds and gts:
        ap50 = compute_ap(preds, gts, 0.5)
        ap5095 = np.mean([compute_ap(preds, gts, t) for t in np.arange(0.5, 1.0, 0.05)])
        print(f'  coords={use_coords} val AP50={ap50:.3f} mAP50-95={ap5095:.3f}')


def main(use_coords):
    # On Windows, DataLoader workers (num_workers>0) re-import this module;
    # one-time setup and the loop live under the __main__ guard.
    random.seed(1337); np.random.seed(1337); torch.manual_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = f"model_coords_{use_coords}.pth"
    print(f'Using device: {device} | USE_COORDS={use_coords}')
    train_loader = DataLoader(CircleDataset('train'), batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(CircleDataset('val'), batch_size=BATCH,
                            shuffle=False, num_workers=4)

    # Fair ablation: build the control base first (with the same seed, so its
    # shared layers have exactly the same initial values as a control run).
    # Then build the treatment and copy those shared weights, zero-initializing
    # only the two extra coordinate channels. This isolates the coordinate
    # channels as the single variable; otherwise the extra channels would
    # shift RNG initialization for every later layer.
    if use_coords:
        base = SimpleYOLO_DSConv_Coord(S=S, use_coords=False).to(device)
    model = SimpleYOLO_DSConv_Coord(S=S, use_coords=use_coords).to(device)
    if use_coords:
        base_sd, sd = base.state_dict(), model.state_dict()
        with torch.no_grad():
            for k in sd:
                if k == 'features.0.weight':
                    sd[k][:, :3].copy_(base_sd[k])
                    sd[k][:, 3:].zero_()
                else:
                    sd[k].copy_(base_sd[k])
        model.load_state_dict(sd)
        del base
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    mse = nn.MSELoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for imgs, targets, masks in train_loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = forward_loss(model, mse, imgs, targets, masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}')
            validate(model, use_coords, val_loader, device)

    torch.save(model.state_dict(), weights)
    print(f'Saved {weights}')


if __name__ == '__main__':
    main(USE_COORDS)
