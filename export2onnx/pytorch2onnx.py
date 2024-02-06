import torch
from network.crnn import CRNN
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def export_to_onnx(model_path, onnx_path, dynamic=False):
    nclass = 20001
    batch = 1
    height = 32
    width = 480
    
    model = CRNN(32, 1, nclass, 256)
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        print("cuda is True")
        model.cuda()
        dummy_input = torch.randn(batch, 1, height, width, device='cuda')
        # dummy_input.to(device='cuda')
    model.eval()
    input_names = ['input_0']
    output_names = ['output_0']
    if not dynamic:
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
    else:
        dynamic_axes = {input_names[0]: {3: 'int_width'},
                        output_names[0]: {0: 'length'}}
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=12, verbose=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

import sys
import onnx 
def check_model(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

if __name__ == '__main__':
    model_path = 'models/ocr_rec_ratio_15.pth'
    onnx_path = 'models/ocr_rec_ratio_15_dynamic.onnx'
    dynamic = True
    export_to_onnx(model_path, onnx_path, dynamic=dynamic)
    # check_model(onnx_path)

