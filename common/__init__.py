"""Shared plumbing for NoobTorch 1.0. Each lesson imports these instead of
copy-pasting boilerplate, so the new concept in each lesson stays visible."""
from .seeds import set_seed, get_device
from .data import get_mnist_loaders, get_cifar10_loaders
from .engine import train_one_epoch, evaluate, fit
from .metrics import confusion_matrix, per_class_precision_recall, plot_curves
