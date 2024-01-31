import tensorrt as trt
def onnx_to_trt(onnx_path, trt_path, im, verbose=True, workspace=4, dynamic=False):
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError('failed to load ONNX file: {}'.format(onnx_path))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1 << 30)
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    # outputs = [network.get_output(i) for i in range(network.num_outputs)]

    # for inp in inputs:
    #     if not verbose:
    #         print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    #     # LOGGER.info(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    # for out in outputs:
    #     if not verbose:
    #         print(f'output "{out.name}" with shape{out.shape} {out.dtype}')
    #     # LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
    if dynamic:
        print("dynamic...")
        if im.shape[0] <= 1:
            print("WARNING --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        # for inp in inputs:
            # profile.set_shape(inp.name, (int(1), *im.shape[1:-1], int(im.shape[-1]/2)), im.shape, (int(8), *im.shape[1:-1], int(im.shape[-1]*2)))
        profile.set_shape('input_0', (1, 1, 32, 128), (1, 1, 32, 240), (1, 1, 32, 480))
        config.add_optimization_profile(profile)
    with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as t:
        t.write(engine.serialize())
    # serialized_engine = builder.build_serialized_network(network, config)
    # with open(trt_path, 'wb') as f:
    #     f.write(serialized_engine)

import torch
if __name__ == '__main__':
    onnx_path = 'models/ocr_rec_ratio_15_dynamic.onnx'
    trt_path = 'models/ocr_rec_ratio_15_dynamic.engine'
    dynamic = True
    dummy_input = torch.randn(1, 1, 32, 480, device='cuda')
    onnx_to_trt(onnx_path, trt_path, im=dummy_input, verbose=False, dynamic=dynamic)