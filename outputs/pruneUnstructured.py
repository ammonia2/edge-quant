import argparse
import torch.nn.utils.prune as prune
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--finalSparsity', type=float, default=0.8)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--epochsPerIter', type=int, default=2)
parser.add_argument('--finalEpochs', type=int, default=3)
args = parser.parse_args()

setSeed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainLoader = getTrainLoader()
testImages, testLabels = getTestTensors()

model = loadModel('fp32').to(device)
targets = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

# iterative global magnitude pruning: prune a bit, fine-tune, repeat (masks stay in place)
prevSparsity = 0.0
for i in range(1, args.iterations + 1):
    sparsity = args.finalSparsity * i / args.iterations
    amount = (sparsity - prevSparsity) / (1 - prevSparsity)  # fraction of the weights still alive
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    prevSparsity = sparsity
    print(f'iteration {i}/{args.iterations}  sparsity {getSparsity(model) * 100:.1f}%')
    trainModel(model, trainLoader, testImages, testLabels, args.epochsPerIter, 0.01, device)

# last fine-tuning round to recover accuracy
trainModel(model, trainLoader, testImages, testLabels, args.finalEpochs, 0.01, device)

# make pruning permanent: weight = weight_orig * mask, then drop the mask
for module, name in targets:
    prune.remove(module, name)

saveModel(model, paths['unstructured'])
print(f"saved {paths['unstructured']}  sparsity {getSparsity(model) * 100:.1f}%  "
      f"testAcc {accuracy(model, testImages, testLabels, device):.2f}")
