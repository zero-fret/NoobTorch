"""The training loop, written once.

Every classification lesson calls fit() so the readme can focus on the
single new idea instead of re-explaining the same 40 lines of boilerplate.
"""
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Returns (average_loss, accuracy_percent) for one pass over the data."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (average_loss, accuracy_percent, predictions, labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        preds.append(pred.cpu())
        targets.append(labels.cpu())
    return total_loss / total, 100.0 * correct / total, torch.cat(preds), torch.cat(targets)


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs, scheduler=None, verbose=True):
    """Trains for `epochs` epochs and returns a history dict of curves.

    history = {'train_loss': [...], 'train_acc': [...],
               'val_loss':   [...], 'val_acc':   [...], 'lr': [...]}
    """
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr')}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        for k, v in (('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('val_loss', va_loss), ('val_acc', va_acc), ('lr', lr)):
            history[k].append(v)
        if verbose:
            print(f'Epoch {epoch:3d}/{epochs} | '
                  f'train loss {tr_loss:.4f} acc {tr_acc:5.2f}% | '
                  f'val loss {va_loss:.4f} acc {va_acc:5.2f}% | lr {lr:.5f}')
    return history
