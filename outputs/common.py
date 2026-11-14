import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import (QuantStub, DeQuantStub, fuse_modules, fuse_modules_qat,
                                   get_default_qat_qconfig, prepare_qat)

backend = 'fbgemm'  # x86 CPUs (Colab). Use 'qnnpack' on ARM.
torch.backends.quantized.engine = backend

classNames = ('airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

paths = {
    'fp32': 'checkpoints/fp32.pth',
    'ptq': 'checkpoints/ptqInt8.pt',
    'qat': 'checkpoints/qatInt8.pt',
    'unstructured': 'checkpoints/unstructured.pth',
    'structured': 'checkpoints/structured.pth',
}
modelNames = list(paths)

for folder in ['checkpoints', 'results', 'onnx', 'data']:
    os.makedirs(folder, exist_ok=True)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
trainTf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                     T.ToTensor(), T.Normalize(mean, std)])
testTf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def setSeed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- data ----------------

def getTrainLoader(batchSize=128):
    trainSet = CIFAR10(root='data', train=True, download=True, transform=trainTf)
    return DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=2)


def getCalibLoader(numImages=2000):
    # training images without augmentation, fixed random subset
    trainSet = CIFAR10(root='data', train=True, download=True, transform=testTf)
    idx = torch.randperm(len(trainSet), generator=torch.Generator().manual_seed(0))[:numImages]
    return DataLoader(Subset(trainSet, idx.tolist()), batch_size=100)


def getTestTensors():
    testSet = CIFAR10(root='data', train=False, download=True, transform=testTf)
    batches = list(DataLoader(testSet, batch_size=1000))
    images = torch.cat([b[0] for b in batches])
    labels = torch.cat([b[1] for b in batches])
    return images, labels


# ---------------- model ----------------

class BasicBlock(nn.Module):
    def __init__(self, inCh, outCh, stride, midCh):
        super().__init__()
        self.conv1 = nn.Conv2d(inCh, midCh, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(midCh)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(midCh, outCh, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outCh)
        if stride == 1 and inCh == outCh:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(inCh, outCh, 1, stride, bias=False),
                                          nn.BatchNorm2d(outCh))
        self.add = FloatFunctional()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add.add(out, self.shortcut(x))
        return self.relu2(out)


class Resnet18(nn.Module):
    # midChannels = width inside each block (conv1 output / conv2 input), used by structured pruning
    def __init__(self, midChannels=None, numClasses=10):
        super().__init__()
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
               (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        self.midChannels = midChannels or [c[1] for c in cfg]
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(*[BasicBlock(i, o, s, m)
                                      for (i, o, s), m in zip(cfg, self.midChannels)])
        self.fc = nn.Linear(512, numClasses)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return self.dequant(x)


def fuseModel(model, qat=False):
    fuse = fuse_modules_qat if qat else fuse_modules  # qat: model must be in train mode, ptq: eval mode
    fuse(model, [['conv1', 'bn1', 'relu']], inplace=True)
    for block in model.blocks:
        fuse(block, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
        if len(block.shortcut) > 0:
            fuse(block.shortcut, [['0', '1']], inplace=True)


def prepareQat(model):
    # fuse, attach the QAT qconfig and insert fake quantize modules (model must be in train mode)
    model.train()
    fuseModel(model, qat=True)
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model


# ---------------- save / load ----------------

def saveModel(model, path):
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({'midChannels': model.midChannels, 'state': state}, path)


def loadModel(name):
    path = paths[name]
    if name in ('ptq', 'qat'):  # TorchScript INT8 models
        torch.backends.quantized.engine = backend
        return torch.jit.load(path, map_location='cpu').eval()
    ckpt = torch.load(path, map_location='cpu')
    model = Resnet18(ckpt['midChannels'])
    model.load_state_dict(ckpt['state'])
    return model.eval()


def exportOnnx(model, path):
    model = model.cpu().eval()
    torch.onnx.export(model, torch.randn(1, 3, 32, 32), path,
                      input_names=['input'], output_names=['logits'],
                      dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=13, dynamo=False)


# ---------------- train / eval ----------------

@torch.no_grad()
def predictAll(model, images, batchSize=128, device='cpu'):
    model.eval()
    preds = [model(images[i:i + batchSize].to(device)).argmax(1).cpu()
             for i in range(0, len(images), batchSize)]
    return torch.cat(preds)


def accuracy(model, images, labels, device='cpu'):
    return (predictAll(model, images, device=device) == labels).float().mean().item() * 100


def trainModel(model, trainLoader, testImages, testLabels, epochs, lr, device, epochHook=None):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        if epochHook:
            epochHook(epoch)
        model.train()
        totalLoss = 0
        for images, labels in trainLoader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        scheduler.step()
        acc = accuracy(model, testImages, testLabels, device)
        print(f'epoch {epoch + 1}/{epochs}  loss {totalLoss / len(trainLoader):.4f}  testAcc {acc:.2f}')
    return model


# ---------------- model stats ----------------

def countParams(model):
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    nonZero = sum((p != 0).sum().item() for p in params)
    return total, nonZero


def getSparsity(model):
    weights = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    zeros = sum((w == 0).sum().item() for w in weights)
    return zeros / sum(w.numel() for w in weights)
