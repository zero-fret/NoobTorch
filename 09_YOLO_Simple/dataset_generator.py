import os
import cv2
import random
from glob import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_num, val_num = 12800, 3200
data_dir = "data"


def find_bg_dir():
    """Use a local ./background if you copied photos here, otherwise the
    shared assets/backgrounds at the repo root."""
    for cand in ("background", "../assets/backgrounds"):
        if os.path.isdir(cand) and glob(f"{cand}/*.jpg"):
            return cand
    return "background"


bg_dir = find_bg_dir()
for split in ("train", "val"):
    os.makedirs(f"{data_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{data_dir}/{split}/labels", exist_ok=True)

bgs = glob(f"{bg_dir}/*.png") + glob(f"{bg_dir}/*.jpg") + glob(f"{bg_dir}/*.jpeg")
assert bgs, f"No background images found in {bg_dir}"


def gen(save_path):
    img = cv2.resize(cv2.imread(random.choice(bgs)), (640, 640))
    r = random.randint(10, 20)
    x = random.randint(r, 640 - r)
    y = random.randint(r, 640 - r)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if random.choice([True, False]):   # ring
        thickness = random.randint(int(r / 3), int(2 * r / 3))
        cv2.circle(img, (x, y), int(r - thickness / 2), color, thickness)
    else:                              # filled disk
        cv2.circle(img, (x, y), r, color, -1)
    cv2.imwrite(save_path, img)
    with open(save_path.replace("images", "labels").replace(".png", ".txt"), "w") as f:
        f.write(f"0 {x/640:.6f} {y/640:.6f} {2*r/640:.6f} {2*r/640:.6f}")


for i in range(train_num):
    gen(f"{data_dir}/train/images/{i:08d}.png")
for i in range(val_num):
    gen(f"{data_dir}/val/images/{i:08d}.png")
print(f"Done. {train_num} train, {val_num} val images (backgrounds: {bg_dir}).")
