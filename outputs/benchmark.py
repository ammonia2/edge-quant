import argparse
import pandas as pd
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--numImages', type=int, default=10000)
args = parser.parse_args()

images, labels = getTestTensors()
images, labels = images[:args.numImages], labels[:args.numImages]
print(f'cpu threads: {torch.get_num_threads()}, test images: {len(images)}')

summary = []
predictions = pd.DataFrame({'index': range(len(labels)), 'trueLabel': labels.tolist()})

for name in modelNames:
    if not os.path.exists(paths[name]):
        print(f'{name}: checkpoint missing, skipped')
        continue
    model = loadModel(name)

    # latency: one image at a time, after a short warm-up
    with torch.no_grad():
        for i in range(20):
            model(images[i:i + 1])
        start = time.perf_counter()
        for i in range(200):
            model(images[i:i + 1])
        latencyMs = (time.perf_counter() - start) / 200 * 1000

    # throughput + predictions: whole test set in batches
    start = time.perf_counter()
    preds = predictAll(model, images)
    seconds = time.perf_counter() - start
    predictions[name + 'Pred'] = preds.tolist()

    row = {'model': name,
           'accuracyPct': round((preds == labels).float().mean().item() * 100, 2),
           'sizeMB': round(os.path.getsize(paths[name]) / 1e6, 2),
           'latencyMsBatch1': round(latencyMs, 3),
           'throughputImgPerSec': round(len(images) / seconds, 1)}
    if name not in ('ptq', 'qat'):  # packed int8 weights have no regular parameters to count
        total, nonZero = countParams(model)
        row.update({'params': total, 'nonZeroParams': nonZero,
                    'sparsityPct': round(getSparsity(model) * 100, 2)})
    summary.append(row)
    print(row)

pd.DataFrame(summary).to_csv('results/benchmark.csv', index=False)
predictions.to_csv('results/predictions.csv', index=False)
print('saved results/benchmark.csv and results/predictions.csv')
