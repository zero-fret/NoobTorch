import os
import cv2
import numpy as np
import random
from glob import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_num, val_num = 3200, 800
bg_dir = "background"
data_dir = "data"
os.makedirs(f"{data_dir}/train/images", exist_ok=True)
os.makedirs(f"{data_dir}/train/labels", exist_ok=True)
os.makedirs(f"{data_dir}/val/images", exist_ok=True)
os.makedirs(f"{data_dir}/val/labels", exist_ok=True)

bgs = glob(f"{bg_dir}/*.png") + glob(f"{bg_dir}/*.jpg") + glob(f"{bg_dir}/*.jpeg")

def gen(img_path, save_path):
    img = cv2.resize(cv2.imread(img_path), (640, 640))
    h, w = 640, 640
    r = random.randint(20, 60)
    x, y = random.randint(r, w-r), random.randint(r, h-r)
    bgr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if random.choice([True, False]):
        thickness = random.randint(int(r/3), int(2*r/3))
        cv2.circle(img, (x, y), int(r-thickness/2), bgr, thickness)
    else:
        cv2.circle(img, (x, y), r, bgr, -1)  
    
    cv2.imwrite(save_path, img)
    with open(save_path.replace("images", "labels").replace(".png", ".txt"), "w") as f:
        f.write(f"0 {x/w:.6f} {y/h:.6f} {2*r/w:.6f} {2*r/h:.6f}")

for i in range(train_num):
    gen(random.choice(bgs), f"{data_dir}/train/images/{i:08d}.png")
    print(i)
for i in range(val_num):
    gen(random.choice(bgs), f"{data_dir}/val/images/{i:08d}.png")
    print(i)
print(f"Done! Generated {train_num} train and {val_num} val images.")