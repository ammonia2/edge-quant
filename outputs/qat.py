import argparse
from torch.ao.quantization import convert, disable_observer
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=4)
args = parser.parse_args()

setSeed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainLoader = getTrainLoader()
testImages, testLabels = getTestTensors()

# 1. start from the trained fp32 model, fuse and insert fake quantize modules
model = prepareQat(loadModel('fp32'))


# 2. fine-tune; stop updating observers and batchnorm statistics in the last epochs
def epochHook(epoch):
    if epoch >= 2:
        model.apply(disable_observer)
    if epoch >= 3:
        model.apply(freeze_bn_stats)


trainModel(model, trainLoader, testImages, testLabels, args.epochs, 0.001, device, epochHook)

# 3. convert to int8 (cpu only) and save as TorchScript
model.cpu().eval()
torch.save(model.state_dict(), 'checkpoints/qatFakeQuant.pth')  # used for the onnx export
int8Model = convert(model)
traced = torch.jit.trace(int8Model, torch.randn(1, 3, 32, 32))
traced.save(paths['qat'])

print(f"saved {paths['qat']}  testAcc {accuracy(loadModel('qat'), testImages, testLabels):.2f}")
