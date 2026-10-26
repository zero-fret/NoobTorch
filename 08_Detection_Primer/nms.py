"""Non-Maximum Suppression: keep one box per object."""


def nms(boxes, iou_thresh):
    """boxes: list of [x1, y1, x2, y2, score].
    Greedily keep the highest-scoring box and drop any lower-scoring box
    that overlaps it by more than iou_thresh (a duplicate of the same object).
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    from boxes import iou
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best[:4], b[:4]) < iou_thresh]
    return keep
