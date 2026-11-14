import argparse
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.5)  # fraction of channels removed
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


def pickChannels(convWeight, ratio):
    # keep the filters with the largest L1 norm
    importance = convWeight.abs().sum(dim=(1, 2, 3))
    numKeep = max(1, int(len(importance) * (1 - ratio)))
    return importance.topk(numKeep).indices.sort().values


def buildPrunedModel(model, ratio):
    # In every residual block only the channels between conv1 and conv2 are pruned.
    # The block input/output channels are tied to the skip connection, so they stay untouched.
    keep = [pickChannels(block.conv1.weight, ratio) for block in model.blocks]
    newModel = Resnet18([len(k) for k in keep])

    oldState = model.state_dict()
    newState = newModel.state_dict()
    for key in newState:
        if oldState[key].shape == newState[key].shape:
            newState[key] = oldState[key]  # layers whose shape did not change

    for i, idx in enumerate(keep):
        prefix = f'blocks.{i}.'
        newState[prefix + 'conv1.weight'] = oldState[prefix + 'conv1.weight'][idx]  # drop output filters
        for name in ['weight', 'bias', 'running_mean', 'running_var']:
            newState[prefix + 'bn1.' + name] = oldState[prefix + 'bn1.' + name][idx]
        newState[prefix + 'conv2.weight'] = oldState[prefix + 'conv2.weight'][:, idx]  # drop matching inputs

    newModel.load_state_dict(newState)
    return newModel


if __name__ == '__main__':
    setSeed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainLoader = getTrainLoader()
    testImages, testLabels = getTestTensors()

    model = loadModel('fp32')
    prunedModel = buildPrunedModel(model, args.ratio)
    print('midChannels:', model.midChannels, '->', prunedModel.midChannels)
    print('params:', countParams(model)[0], '->', countParams(prunedModel)[0])
    print(f'testAcc right after pruning {accuracy(prunedModel, testImages, testLabels):.2f}')

    trainModel(prunedModel, trainLoader, testImages, testLabels, args.epochs, 0.01, device)

    saveModel(prunedModel, paths['structured'])
    print(f"saved {paths['structured']}  testAcc {accuracy(prunedModel, testImages, testLabels, device):.2f}")
