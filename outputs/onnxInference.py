import argparse
import onnxruntime as ort
import pandas as pd
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--numImages', type=int, default=10000)
args = parser.parse_args()

onnxNames = ['fp32', 'ptqInt8', 'qatInt8', 'unstructured', 'structured']

images, labels = getTestTensors()
images, labels = images[:args.numImages].numpy(), labels[:args.numImages]

rows = []
for name in onnxNames:
    path = f'onnx/{name}.onnx'
    if not os.path.exists(path):
        print(f'{name}: onnx file missing, skipped')
        continue
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

    for i in range(20):  # warm-up
        session.run(None, {'input': images[i:i + 1]})
    start = time.perf_counter()
    for i in range(200):
        session.run(None, {'input': images[i:i + 1]})
    latencyMs = (time.perf_counter() - start) / 200 * 1000

    start = time.perf_counter()
    preds = np.concatenate([session.run(None, {'input': images[i:i + 100]})[0].argmax(1)
                            for i in range(0, len(images), 100)])
    seconds = time.perf_counter() - start

    row = {'model': name,
           'accuracyPct': round((preds == labels.numpy()).mean() * 100, 2),
           'sizeMB': round(os.path.getsize(path) / 1e6, 2),
           'latencyMsBatch1': round(latencyMs, 3),
           'throughputImgPerSec': round(len(images) / seconds, 1)}
    rows.append(row)
    print(row)

pd.DataFrame(rows).to_csv('results/onnxBenchmark.csv', index=False)
print('saved results/onnxBenchmark.csv')
