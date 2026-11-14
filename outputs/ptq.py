from torch.ao.quantization import get_default_qconfig, prepare, convert
from common import *

numCalibImages = 2000

setSeed(0)
model = loadModel('fp32')  # cpu, eval mode

# 1. fuse conv + bn + relu
fuseModel(model)

# 2. backend config + observers
model.qconfig = get_default_qconfig(backend)
prepare(model, inplace=True)

# 3. calibration on a subset of the training data
with torch.no_grad():
    for images, _ in getCalibLoader(numCalibImages):
        model(images)

# 4. convert to int8 and save as a TorchScript file (loads without the model class)
int8Model = convert(model)
traced = torch.jit.trace(int8Model, torch.randn(1, 3, 32, 32))
traced.save(paths['ptq'])

testImages, testLabels = getTestTensors()
print(f"saved {paths['ptq']}  testAcc {accuracy(loadModel('ptq'), testImages, testLabels):.2f}")
