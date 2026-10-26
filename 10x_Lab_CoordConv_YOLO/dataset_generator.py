import os
import cv2
import random
from glob import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_num, val_num = 3200, 800
data_dir = "data"


def find_bg_dir():
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

# Four circles, one confined to each quadrant -> 4 positive cells per image.
corners = [
    (lambda r: r,                 lambda r: 320 - r - 5,
     lambda r: r,                 lambda r: 320 - r - 5),
    (lambda r: 320 + r + 5,       lambda r: 640 - r,
     lambda r: r,                 lambda r: 320 - r - 5),
    (lambda r: r,                 lambda r: 320 - r - 5,
     lambda r: 320 + r + 5,       lambda r: 640 - r),
    (lambda r: 320 + r + 5,       lambda r: 640 - r,
     lambda r: 320 + r + 5,       lambda r: 640 - r),
]


def gen(save_path):
    img = cv2.resize(cv2.imread(random.choice(bgs)), (640, 640))
    radii = [random.randint(10, 15) for _ in range(4)]
    lines = []
    for r, (xlo, xhi, ylo, yhi) in zip(radii, corners):
        x = random.randint(xlo(r), max(xhi(r), xlo(r)))
        y = random.randint(ylo(r), max(yhi(r), ylo(r)))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if random.choice([True, False]):
            t = random.randint(int(r / 3), int(2 * r / 3))
            cv2.circle(img, (x, y), int(r - t / 2), color, t)
        else:
            cv2.circle(img, (x, y), r, color, -1)
        lines.append(f"0 {x/640:.6f} {y/640:.6f} {2*r/640:.6f} {2*r/640:.6f}")
    cv2.imwrite(save_path, img)
    with open(save_path.replace("images", "labels").replace(".png", ".txt"), "w") as f:
        f.write("\n".join(lines))


for i in range(train_num):
    gen(f"{data_dir}/train/images/{i:08d}.png")
for i in range(val_num):
    gen(f"{data_dir}/val/images/{i:08d}.png")
print(f"Done. {train_num} train, {val_num} val images, 4 circles each (backgrounds: {bg_dir}).")
