import torch
from network.crnn import *
from utils.alphabets import alphabet
from utils.help import *

class CRNNInfer():
    def __init__(self, model_path) -> None:
        nclass = 20001
        crnn_model = CRNN(32, 1, nclass, 256)
        if torch.cuda.is_available():
            crnn_model = crnn_model.cuda()
            crnn_model.eval()
        crnn_model.load_state_dict(torch.load(model_path))
        self.crnn_model = crnn_model
        self.alphabet = alphabet
        self.converter = strLabelConverter(alphabet)
        self.resize = resizeNormalize()

    def predict(self, img):
        im = pad_rec_im(img, wh_ratio=15)
        pred, conf = crnn_rec(im, self.crnn_model, self.converter, [True, self.resize])
        return pred, conf

import onnxruntime as ort
from pdb import set_trace
def onnx_infer(img):
    img = pad_rec_im(img, wh_ratio=15)
    rec_im = Image.fromarray(img)
    size = (int(1. * rec_im.size[0] * 32 / rec_im.size[1]), 32)
    # size = (480, 32)
    rec_im = rec_im.convert('L')
    transformer = resizeNormalize()
    img = transformer(rec_im, size)
    img = np.expand_dims(img, axis=0)

    converter = strLabelConverter(alphabet)
    model_path = 'models/ocr_rec_ratio_15_dynamic.onnx'
    sess = ort.InferenceSession(model_path)
    # sess.set_providers([''])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(input_name, output_name)
    output = sess.run([output_name], {input_name: img})
    output = output[0]
    
    preds = torch.from_numpy(output)
    probs = torch.max(preds, dim=2).values
    conf = torch.exp(torch.sum(probs, dim=0))
    _, predss = preds.max(2)
    preds = predss.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([predss.size(0)] * predss.size(1)))
    pred = converter.decode(preds.data, preds_size.data, True, raw=False)

    return pred, conf

from utils.rt_infer import Rt_Infer
def trt_infer(img):
    trt_path = 'models/ocr_rec_ratio_15_dynamic.engine'
    # trt_path = 'models/ocr_rec_ratio_15.engine'
    # trt_path = 'models/ocr_rec_ratio_bat_15.engine'
    crnn_model = Rt_Infer(trt_path)
    if torch.cuda.is_available():
        crnn_model = crnn_model.cuda().eval()
    im = pad_rec_im(img, wh_ratio=12)
    converter = strLabelConverter(alphabet)
    resize = resizeNormalize()
    pred, conf = crnn_rec(im, crnn_model, converter, (True, resize))
    # set_trace()
    # print(pred)
    return pred, conf

import cv2
if __name__ == '__main__':
    # model_path = 'models/ocr_rec_ratio_15.pth'
    # crnn = CRNNInfer(model_path)
    img_path = 'data/rec_1.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pred, conf = crnn.predict(img)
    # print(pred)
    # pred, conf = onnx_infer(img)
    pred, conf = trt_infer(img)
    print(pred)
   

