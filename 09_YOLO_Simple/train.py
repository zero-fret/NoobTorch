import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import CircleDataset, S, IMG_SIZE, decode_predictions, nms, compute_ap, decode_gt
from model import SimpleYOLO

BATCH = 32
EPOCHS = 120
LR = 0.005
COORD_W, CONF_W, NOOBJ_W = 1.0, 3.0, 0.5


def forward_loss(model, mse, bce, imgs, targets, masks):
    out = model(imgs)
    pred_tx = torch.sigmoid(out[:, 0:1]) * 2 - 0.5
    pred_ty = torch.sigmoid(out[:, 1:2]) * 2 - 0.5
    pred_tw = torch.sigmoid(out[:, 2:3])
    pred_th = torch.sigmoid(out[:, 3:4])
    conf_logits = out[:, 4:5]

    coord = mse(pred_tx, targets[:, 0:1]) + mse(pred_ty, targets[:, 1:2]) \
          + mse(pred_tw, targets[:, 2:3]) + mse(pred_th, targets[:, 3:4])
    mask = masks.unsqueeze(1)
    coord = (coord * mask.float()).sum() / (mask.sum() + 1e-6) * COORD_W

    conf_loss = bce(conf_logits, targets[:, 4:5])
    weight = torch.ones_like(targets[:, 4:5]) * NOOBJ_W
    weight = weight * (1 - targets[:, 4:5]) + targets[:, 4:5] * CONF_W
    conf_loss = (conf_loss * weight).sum() / (weight.sum() + 1e-6)
    return coord + conf_loss


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    preds, gts = [], []
    for b, (imgs, targets, masks) in enumerate(val_loader):
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            img_id = b * imgs.size(0) + i
            boxes = decode_predictions(model(imgs[i:i+1]), S, conf_thresh=0.01)
            for box in nms(boxes, 0.5):
                preds.append((img_id, box[:4], box[4], 0))
            gts.extend(decode_gt(targets[i], masks[i], img_id, S))
    if preds and gts:
        ap50 = compute_ap(preds, gts, 0.5)
        ap5095 = np.mean([compute_ap(preds, gts, t) for t in np.arange(0.5, 1.0, 0.05)])
        print(f'  val AP50={ap50:.3f} mAP50-95={ap5095:.3f}')


def main():
    # On Windows, DataLoader workers (num_workers>0) re-import this module.
    # Everything that must run once goes under main() behind the __main__
    # guard so workers don't re-run the whole training loop.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_loader = DataLoader(CircleDataset('train'), batch_size=BATCH,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CircleDataset('val'), batch_size=BATCH,
                            shuffle=False, num_workers=4)

    model = SimpleYOLO(S=S).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    mse = nn.MSELoss(reduction='none')
    bce = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for imgs, targets, masks in train_loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = forward_loss(model, mse, bce, imgs, targets, masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}/{EPOCHS} loss={total/len(train_loader)*100:.3f}')
        if (epoch + 1) % 10 == 0:
            validate(model, val_loader, device)

    torch.save(model.state_dict(), 'model.pth')
    print('Saved model.pth')


if __name__ == '__main__':
    main()
