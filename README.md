# Model Optimization Pipeline (PTQ, QAT, Pruning, ONNX) - ResNet-18 on CIFAR-10

## Requirements
- Python 3.9+, PyTorch + torchvision, pandas, pillow, onnx, onnxruntime
- Google Colab: torch, torchvision and pandas are preinstalled; the notebook installs `onnx` and `onnxruntime`.
- Use a GPU runtime (Runtime > Change runtime type > T4 GPU) for training. Quantized conversion, benchmarking and ONNX run on CPU.

## Run in Colab
Open `optimizationPipeline.ipynb` and use Runtime > Run all. Each cell writes one script and then runs it.

## Run manually (same order as the notebook)
```
pip install onnx onnxruntime
python trainBaseline.py                  # fp32 model        -> checkpoints/fp32.pth
python ptq.py                            # PTQ int8          -> checkpoints/ptqInt8.pt
python qat.py                            # QAT int8          -> checkpoints/qatInt8.pt
python pruneUnstructured.py              # unstructured      -> checkpoints/unstructured.pth
python pruneStructured.py                # structured        -> checkpoints/structured.pth
python inference.py --model all          # or --model ptq --image cat.png
python benchmark.py                      # results/benchmark.csv, results/predictions.csv
python exportOnnx.py                     # onnx/*.onnx
python onnxInference.py                  # results/onnxBenchmark.csv
```
Quick test run: `python trainBaseline.py --epochs 1`, `python qat.py --epochs 1`, etc.

## Files
| File | Purpose |
|---|---|
| common.py | model, data, train/eval helpers, save/load |
| trainBaseline.py | trains the FP32 ResNet-18 |
| ptq.py | fuse, calibrate on 2000 train images, convert to INT8 |
| qat.py | fuse, insert fake quant, fine-tune, convert to INT8 |
| pruneUnstructured.py | iterative global L1 pruning + fine-tuning, makes masks permanent |
| pruneStructured.py | removes 50% of block-internal channels, rebuilds a smaller model, fine-tunes |
| inference.py | predicts with any of the five models |
| benchmark.py | accuracy, size, latency, throughput, per-image predictions |
| exportOnnx.py / onnxInference.py | ONNX export and CPU inference with onnxruntime |

## Notes
- INT8 PyTorch models are TorchScript files (`torch.jit.load`), no model class needed. Backend is `fbgemm` (x86); change `backend` in common.py to `qnnpack` on ARM.
- Unstructured pruning keeps dense tensors with zeros, so file size and CPU speed do not improve. Structured pruning gives a really smaller model.
- Structured pruning only removes channels between conv1 and conv2 inside each residual block, because the other channels are tied to the skip connections.
- ONNX PTQ INT8 is produced with onnxruntime static quantization on the FP32 ONNX model. ONNX QAT INT8 is the fake-quant graph exported as QuantizeLinear/DequantizeLinear nodes.
