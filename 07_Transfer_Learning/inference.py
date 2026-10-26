"""Load a trained model and run it on data -- the thing deployment actually is.

Edit the two constants below:
    IMAGE_PATH = None   -> classify the first CIFAR test image
    IMAGE_PATH = 'cat.jpg' -> classify your own image
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from common.data import IMAGENET_MEAN, IMAGENET_STD, DATA_ROOT

# Point this at your own image, or leave None to classify a CIFAR test image.
IMAGE_PATH = None
WEIGHTS_PATH = 'cifar10_resnet18.pth'   # or 'cifar10_resnet18_finetuned.pth'

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(weights_path):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def predict(model, pil_image):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(pil_image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return CLASSES[idx], conf.item(), probs


if __name__ == '__main__':
    model = build_model(WEIGHTS_PATH)

    if IMAGE_PATH is not None:
        img = Image.open(IMAGE_PATH)
    else:
        test_ds = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=False, download=True)
        img, label = test_ds[0]
        print(f'(CIFAR test image, true label: {CLASSES[label]})')

    name, conf, probs = predict(model, img)
    print(f'Prediction: {name} ({conf:.1%})')
    top3 = probs.topk(3)
    for p, i in zip(top3.values, top3.indices):
        print(f'  {CLASSES[i]:>12}: {p:.1%}')
