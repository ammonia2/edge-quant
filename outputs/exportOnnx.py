from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver, QConfig,
                                   default_per_channel_weight_fake_quant, prepare_qat)
from common import *


class CalibReader(CalibrationDataReader):
    def __init__(self, numImages=500):
        self.batches = iter([images.numpy() for images, _ in getCalibLoader(numImages)])

    def get_next(self):
        batch = next(self.batches, None)
        return None if batch is None else {'input': batch}


# float models: fp32, unstructured pruned, structured pruned
for name in ['fp32', 'unstructured', 'structured']:
    exportOnnx(loadModel(name), f'onnx/{name}.onnx')
    print(f'exported onnx/{name}.onnx')

# PTQ int8: static quantization of the fp32 onnx model with onnxruntime (same calibration idea as ptq.py)
quantize_static('onnx/fp32.onnx', 'onnx/ptqInt8.onnx', CalibReader(),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
print('exported onnx/ptqInt8.onnx')

# QAT int8: plain FakeQuantize modules (the fused training op and the 0..127 range are not exportable)
exportQconfig = QConfig(
    activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                      dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=default_per_channel_weight_fake_quant)
qatModel = Resnet18().train()
fuseModel(qatModel, qat=True)
qatModel.qconfig = exportQconfig
prepare_qat(qatModel, inplace=True)
qatModel.load_state_dict(torch.load('checkpoints/qatFakeQuant.pth'))
exportOnnx(qatModel, 'onnx/qatInt8.onnx')
print('exported onnx/qatInt8.onnx')
