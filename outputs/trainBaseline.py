import argparse
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

setSeed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainLoader = getTrainLoader()
testImages, testLabels = getTestTensors()

model = trainModel(Resnet18(), trainLoader, testImages, testLabels, args.epochs, 0.1, device)
saveModel(model, paths['fp32'])
print(f"saved {paths['fp32']}  testAcc {accuracy(model, testImages, testLabels, device):.2f}")
