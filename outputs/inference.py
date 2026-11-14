import argparse
from PIL import Image
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='all', choices=['all'] + modelNames)
parser.add_argument('--image', default=None)  # path to any image, otherwise 5 CIFAR-10 test images are used
args = parser.parse_args()


def predict(model, images):
    with torch.no_grad():
        probs = F.softmax(model(images), dim=1)
    confidence, pred = probs.max(1)
    return pred, confidence


if args.image:
    images = testTf(Image.open(args.image).convert('RGB').resize((32, 32))).unsqueeze(0)
    labels = None
else:
    images, labels = getTestTensors()
    images, labels = images[:5], labels[:5]

names = modelNames if args.model == 'all' else [args.model]
for name in names:
    if not os.path.exists(paths[name]):
        print(f'{name}: checkpoint missing, skipped')
        continue
    pred, confidence = predict(loadModel(name), images)
    print(f'--- {name} ---')
    for i in range(len(images)):
        truth = f'  (true: {classNames[labels[i]]})' if labels is not None else ''
        print(f'image {i}: {classNames[pred[i]]}  {confidence[i] * 100:.1f}%{truth}')
