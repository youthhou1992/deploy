import cv2
import torch
import numpy as np
from network.SR import SuperResolutionNet

def init_torch_model():
    torch_model = SuperResolutionNet()
    state_dict = torch.load('models/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth')['state_dict']
    #old_key: generator.conv1.weight
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)
    
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

def main():
    model = init_torch_model()
    input_img = cv2.imread('data/000001.png').astype(np.float32)
    print("input shape:", input_img.shape)
    #(h, w, c)  --> (b, c, h, w)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)
    torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

    #(b, c, h, w) --> (h, w, c)
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)
    print(torch_output.shape)
    cv2.imwrite("data/face_torch_3.png", torch_output)

def main2():
    model = init_torch_model()
    dummy_input = torch.randn(1, 3, 256, 256)
    # factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)
    factor = torch.tensor([1, 1, 3,3], dtype=torch.float)
    with torch.no_grad():
        torch.onnx.export(model, (dummy_input, factor), 'models/srcnn3.onnx', opset_version=11, input_names=['input', 'factor'], output_names=['output'])

def main3():
    import onnxruntime 
    input_img = cv2.imread('data/000001.png').astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    input_factor = np.array([1, 1, 4, 4], dtype=np.float32) 
    ort_session = onnxruntime.InferenceSession("models/srcnn3.onnx") 
    ort_inputs = {'input': input_img, 'factor': input_factor} 
    ort_output = ort_session.run(None, ort_inputs)[0] 
    
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite("data/face_ort_3.png", ort_output) 

if __name__ == '__main__':
    # main()
    # main2()
    main3()