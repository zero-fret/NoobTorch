import os
import torch
from torch.utils.data import DataLoader
from utils import CircleDataset, S, IMG_SIZE, decode_predictions, nms, compute_map
from model import SimpleYOLO
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    S_grid = S

    model = SimpleYOLO(S=S_grid).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=False))
    model.eval()

    val_dataset = CircleDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    conf_thresh = 0.5
    iou_thresh = 0.5
    
    output_dir = 'test_result'
    os.makedirs(output_dir, exist_ok=True)
    box_color = (0, 255, 0)  # 绿色

    all_predictions = []
    all_groundtruths = []

    with torch.no_grad():
        for batch_idx, (img, target, mask) in enumerate(val_loader):
            img_tensor = img.to(device)
            output = model(img_tensor)

            pred_boxes = decode_predictions(output, S_grid, conf_thresh, IMG_SIZE)
            if pred_boxes:
                pred_boxes = nms(pred_boxes, iou_thresh)
            for box in pred_boxes:
                x1, y1, x2, y2, conf = box
                all_predictions.append((batch_idx, [x1, y1, x2, y2], conf, 0))

            # 提取 GT
            mask_np = mask[0].cpu().numpy()
            if mask_np.any():
                grid_y, grid_x = np.where(mask_np)
                for gy, gx in zip(grid_y, grid_x):
                    tx = target[0, 0, gy, gx].item()
                    ty = target[0, 1, gy, gx].item()
                    tw = target[0, 2, gy, gx].item()
                    th = target[0, 3, gy, gx].item()
                    cx = (gx + tx) / S_grid * IMG_SIZE
                    cy = (gy + ty) / S_grid * IMG_SIZE
                    w = tw * IMG_SIZE
                    h = th * IMG_SIZE
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    all_groundtruths.append((batch_idx, [x1, y1, x2, y2], 0))
            
            # 画框并保存
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)  # RGB
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_draw = img_bgr.copy()
            
            for box in pred_boxes:
                x1, y1, x2, y2, conf = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), box_color, 2)
                label = f'{conf:.2f}'
                cv2.putText(img_draw, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            save_path = os.path.join(output_dir, f'image_{batch_idx:04d}.jpg')
            cv2.imwrite(save_path, img_draw)
            print(f"Saved: {save_path}")

    # 计算AP (非插值)
    ap = compute_map(all_predictions, all_groundtruths, iou_thresh=0.5)
    print(f"AP@0.5: {100*ap:.2f}%")
    
    # 可选：计算多个IoU阈值
    print("\n多阈值AP:")
    for iou in [0.5, 0.75, 0.95]:
        ap_iou = compute_map(all_predictions, all_groundtruths, iou_thresh=iou)
        print(f"  AP@{iou:.2f}: {100*ap_iou:.2f}%")

if __name__ == '__main__':
    test()