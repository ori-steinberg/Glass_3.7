import os
import cv2
import torch
import os.path as osp
from time import time
from typing import List
from torch.quantization import quantize_dynamic_jit
from torch.quantization import per_channel_dynamic_qconfig
from glass_det.glass_main.glass.inference.glass_runner import GlassRunner
from glass_det.glass_main.glass.modeling.recognition.prediction_aster import device
from detectron2.structures import ImageList


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    See get_caffe2_inputs() below.
    """
    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # NOTE: The scale inside im_info is kept as convention and for providing
        # post-processing information if further processing is needed. For
        # current Caffe2 model definitions that don't include post-processing inside
        # the model, this number is not used.
        # NOTE: There can be a slight difference between width and height
        # scales, using a single number can results in numerical difference
        # compared with D2's post-processing.
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)

def compare_model_size(origin_model, quantized_model):
    print("Size of model before quantization")
    print_size_of_model(origin_model)
    print("Size of model after quantization")
    print_size_of_model(quantized_model)

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def model_test(loaded_model):
    plot = test_param['plot']
    config_dir = test_param['config_dir']
    in_path = test_param['in_path']
    out_path = test_param['out_path']
    batch_size = test_param['batch_size']

    glass_model = build_glass_model(config_dir) if plot else None
    files = os.listdir(in_path)[1:16]
    print(f'Found {len(files)} files in {in_path}\nBatch size: {batch_size}, Device: {device.type}')

    time_per_batch = []
    loop_n = len(files) // batch_size
    for i in range(loop_n):
        img_paths = files[i * batch_size:(i + 1) * batch_size]
        data = [cv2.imread(f'{in_path}/{file}') for file in img_paths]
        input_data = [{'image': torch.as_tensor(image.transpose((2, 0, 1)))} for image in data]
        ts = time()
        with torch.no_grad():
            preds = loaded_model(input_data)
            # preds = loaded_model(input_data)
        time_per_batch.append({img_paths[0]: [round(time()) - ts, data[0].shape[:2], len(preds[0][0]['scores'])]})
        if plot:
            # image_shape_list = [image.shape[:2] for image in data]
            image_shape_list = [{name: val for name, val in zip(['origin_height', 'origin_width'], image.shape[:2])} for image in data]
            pred_data = glass_model.post_run(preds, image_shape_list)
            glass_model.plot_img(data, img_paths, out_path, pred_data)
    return time_per_batch

def RunConvertedModel(onnx_file: str, dummy_input: List[dict], ONNX: bool, _device: str = 'cpu') -> None:
    if not ONNX:
        loaded_model = torch.jit.load(onnx_file)
        time_per_batch = model_test(loaded_model)
        return

def ConvertGlassModel2Script(config_path: str, file_save_path: str, dummy_input: List[dict], ONNX: bool, quantisize: bool = False) -> None:
    glass_model = build_glass_model(config_path)
    device = glass_model.model.device
    size_divisibility = glass_model.model.backbone.size_divisibility
    converted_inputs = convert_batched_inputs_to_c2_format(dummy_input, size_divisibility, device)
    # export_tracing(glass_model.model, dummy_input)
    scripted_model = torch.jit.trace(glass_model.model, converted_inputs, check_trace=True, strict=False)
    if quantisize:
        scripted_model = quantize_dynamic_jit(scripted_model, {'': per_channel_dynamic_qconfig})
    torch.jit.save(scripted_model, file_save_path)
    print('Finish Converting model 🥳🥳🥳\nFile location - ', file_save_path)

def build_glass_model(config_dir: str=None):
    model_path = '/Users/ido.nahum/engineCache/Apps/ocr/metadata/assets/glass_textocr.pth'
    config_path = config_dir + ('glass_finetune_textocr_cpu.yaml' if device.type == 'cpu' else 'glass_finetune_textocr_cuda.yaml')
    glass_runner = GlassRunner(model_path=model_path, config_path=config_path, post_process=True)
    return glass_runner


base_path = '/Users/ido.nahum/dev/triton_pytorch/detectron2/glass_det/'
test_param = {
    'plot': True,
    'out_path': base_path + 'compare_dir/GLASS/patches/diageous/quant',
    'config_dir': base_path + 'glass_main/configs/',
    'in_path': base_path + 'compare_dir/data/patches/diageous/',
    'batch_size': 1,
    'sample_path': base_path + '1005741.jpg'
}
if __name__ == '__main__':
    from patches import apply_patches
    apply_patches()
    torch.ops.load_library("/usr/local/Caskroom/miniconda/base/envs/detectron2/lib/python3.8/site-packages/detectron2/_C.cpython-38-darwin.so")
    ONNX = False
    suffix = '.onnx' if ONNX else '.pt'
    model_path = base_path + "glass_model_cpu" + suffix

    imgs = [cv2.imread(test_param['sample_path']), cv2.imread(test_param['sample_path'])]
    dummy_input = [{'image': torch.as_tensor(img.transpose((2, 0, 1)))} for img in imgs]
    ConvertGlassModel2Script(test_param['config_dir'], model_path, dummy_input, ONNX)
    # RunConvertedModel(model_path, dummy_input, ONNX)
