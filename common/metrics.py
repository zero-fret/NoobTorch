"""Evaluation helpers beyond a single accuracy number:
confusion matrix, per-class precision/recall, and learning-curve plots.
"""
import torch


def confusion_matrix(preds, labels, num_classes):
    """cm[true, predicted] = count."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def per_class_precision_recall(cm):
    """Returns (precision, recall, f1) tensors, one value per class."""
    tp = cm.diag().float()
    precision = tp / (cm.sum(dim=0).float() + 1e-12)
    recall = tp / (cm.sum(dim=1).float() + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def plot_curves(history, save_path='results.png'):
    """Saves loss/accuracy curves (train vs val). Needs matplotlib; prints a
    hint instead of crashing if it isn't installed."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed (`pip install matplotlib`) -- skipping plot.')
        return

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='val')
    ax1.set_title('Loss'); ax1.set_xlabel('epoch'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history['train_acc'], label='train')
    ax2.plot(epochs, history['val_acc'], label='val')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('epoch'); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f'Learning curves saved to {save_path}')
