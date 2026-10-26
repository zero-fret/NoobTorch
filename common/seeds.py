"""Reproducibility helper, shared by every lesson.

Same random seed -> same data order / weight init -> same result run to run.
Call set_seed(1337) once at the start of every train script.
"""
import random
import numpy as np
import torch


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """cuda if available, else cpu."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
