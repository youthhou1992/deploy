import torch
import torch.nn as nn
from collections import OrderedDict, namedtuple
import numpy as np
import tensorrt as trt
from pdb import set_trace

class Rt_Infer(nn.Module):
    '''
    description: 基于tensorRT 的 识别模型 推理接口
    '''
    # CRNN tensorrt infer
    def __init__(self, weights='', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO) #trt 记录器 ，记录生成过程的日志
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read()) #读取模型
            #deserialize the engine from a memory buffer 生成engine的过程 为 序列化（serialize）过程，读取engine过程可以看作是反序列化过程
        context = model.create_execution_context() #创建 执行上下文推理任务的引擎
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        self.input_name = 'input_0'
        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            set_trace()
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                self.input_name = name
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        # batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        self.__dict__.update(locals())

    def forward(self, im): 
        # CRNN inference
        # b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            print("fp16...")
            im = im.half()  # to FP16
        # set_trace()
        if self.dynamic and im.shape != self.bindings[self.input_name].shape:
            print("dynamic...")
            i = self.model.get_binding_index(self.input_name)
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=im.shape)
            # for name in self.output_names:
            name = self.output_names[0]
            i = self.model.get_binding_index(name)
            self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        # s = self.bindings['images'].shape
        # assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        # set_trace()
        self.binding_addrs[self.input_name] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # y = [self.bindings[x].data for x in sorted(self.output_names)]
        y = self.bindings[self.output_names[0]].data
        # set_trace()
        return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

if __name__ =="__main__":
    engine_path = r"crnn_w/res.engine"