import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import CircleDataset, S
from model import SimpleYOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    epochs = 50
    lr = 0.001
    S_grid = S

    train_dataset = CircleDataset(split='train')
    val_dataset = CircleDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SimpleYOLO(S=S_grid).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    mse_loss = nn.MSELoss(reduction='none')
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    coord_weight = 5.0
    conf_weight = 1.0
    noobj_weight = 0.5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets, masks in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = model(imgs)   # (B, 5, S, S)

            pred_tx = torch.sigmoid(outputs[:, 0:1, :, :])
            pred_ty = torch.sigmoid(outputs[:, 1:2, :, :])
            pred_tw = torch.sigmoid(outputs[:, 2:3, :, :])
            pred_th = torch.sigmoid(outputs[:, 3:4, :, :])
            pred_conf_logits = outputs[:, 4:5, :, :]

            target_tx = targets[:, 0:1, :, :]
            target_ty = targets[:, 1:2, :, :]
            target_tw = targets[:, 2:3, :, :]
            target_th = targets[:, 3:4, :, :]
            target_conf = targets[:, 4:5, :, :]

            loss_xy = mse_loss(pred_tx, target_tx) + mse_loss(pred_ty, target_ty)
            loss_wh = mse_loss(pred_tw, target_tw) + mse_loss(pred_th, target_th)
            loss_coord = loss_xy + loss_wh
            mask = masks.unsqueeze(1)   # (B,1,S,S)
            loss_coord = (loss_coord * mask.float()).sum() / (mask.sum() + 1e-6) * coord_weight

            loss_conf = bce_loss(pred_conf_logits, target_conf)
            weight = torch.ones_like(target_conf) * noobj_weight
            weight = weight * (1 - target_conf) + target_conf * conf_weight
            loss_conf = (loss_conf * weight).sum() / (weight.sum() + 1e-6)

            loss = loss_coord + loss_conf

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {(avg_loss*100):.4f}, LR: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets, masks in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    masks = masks.to(device)
                    outputs = model(imgs)

                    pred_tx = torch.sigmoid(outputs[:, 0:1, :, :])
                    pred_ty = torch.sigmoid(outputs[:, 1:2, :, :])
                    pred_tw = torch.sigmoid(outputs[:, 2:3, :, :])
                    pred_th = torch.sigmoid(outputs[:, 3:4, :, :])
                    pred_conf_logits = outputs[:, 4:5, :, :]

                    target_tx = targets[:, 0:1, :, :]
                    target_ty = targets[:, 1:2, :, :]
                    target_tw = targets[:, 2:3, :, :]
                    target_th = targets[:, 3:4, :, :]
                    target_conf = targets[:, 4:5, :, :]

                    loss_xy = mse_loss(pred_tx, target_tx) + mse_loss(pred_ty, target_ty)
                    loss_wh = mse_loss(pred_tw, target_tw) + mse_loss(pred_th, target_th)
                    loss_coord = loss_xy + loss_wh
                    mask = masks.unsqueeze(1)
                    loss_coord = (loss_coord * mask.float()).sum() / (mask.sum() + 1e-5) * coord_weight

                    loss_conf = bce_loss(pred_conf_logits, target_conf)
                    weight = torch.ones_like(target_conf) * noobj_weight
                    weight = weight * (1 - target_conf) + target_conf * conf_weight
                    loss_conf = (loss_conf * weight).sum() / (weight.sum() + 1e-6)

                    val_loss += (loss_coord + loss_conf).item()
            val_loss_avg = val_loss / len(val_loader)
            print(f"Validation Loss: {(val_loss_avg*100):.4f}")

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

if __name__ == '__main__':
    train()
