"""Bounding-box geometry for object detection. No neural network here."""

# Box formats used in this course:
#  YOLO label (normalized): (cx, cy, w, h), all in [0,1], fractions of image size
#  Pixel box (xyxy):       (x1, y1, x2, y2), top-left / bottom-right corners
#  Pixel box (cxcywh):     (cx, cy, w, h), center and width/height


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def normalized_to_pixels(box, img_size=640):
    """(cx, cy, w, h) normalized -> (x1, y1, x2, y2) pixel coordinates."""
    cx, cy, w, h = box
    return cxcywh_to_xyxy((cx * img_size, cy * img_size,
                           w * img_size, h * img_size))


def iou(box_a, box_b):
    """Intersection over union of two xyxy boxes. 1 = identical, 0 = no overlap."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def grid_cell_of(cx, cy, S):
    """Which grid cell owns a normalized center (cx, cy). Returns (gx, gy)."""
    gx = min(int(cx * S), S - 1)
    gy = min(int(cy * S), S - 1)
    return gx, gy


def cell_offset(cx, cy, S):
    """Offset of the center from its cell's top-left corner, in [0, 1).
    This is exactly what a YOLO head learns for tx, ty."""
    gx, gy = grid_cell_of(cx, cy, S)
    return cx * S - gx, cy * S - gy


if __name__ == '__main__':
    # A box centered at (320, 320), 40 px wide/tall on a 640 image.
    label = (0.5, 0.5, 40 / 640, 40 / 640)
    xyxy = normalized_to_pixels(label)
    print('normalized -> pixels:', xyxy)

    # Identical boxes -> IoU 1; a shifted twin -> smaller IoU.
    twin = (xyxy[0] + 10, xyxy[1], xyxy[2] + 10, xyxy[3])
    print('IoU identical:', round(iou(xyxy, xyxy), 3))
    print('IoU shifted 10px:', round(iou(xyxy, twin), 3))

    # Grid assignment.
    gx, gy = grid_cell_of(0.5, 0.5, S=20)
    off_x, off_y = cell_offset(0.5, 0.5, S=20)
    print(f'center (0.5,0.5) -> cell ({gx},{gy}), offset ({off_x:.2f},{off_y:.2f})')
