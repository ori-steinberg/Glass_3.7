import os
import cv2
import torch
import logging
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from glass.utils.visualizer import visualize
from glass.utils.common_utils import rgb2grey
from glass.postprocess import build_post_processor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, RotatedBoxes, ImageList
from glass.modeling.recognition.text_encoder import TextEncoder
from glass.config import add_e2e_config, add_glass_config, add_dataset_config, add_post_process_config


class GlassRunner:

    def __init__(self, model_path: str, config_path: str, opts: List[str] = None, post_process=True):
        """
        Initializes a runner that can run inference using a detectron2 based model
        Args:
            model_path: Path to the detectron2 model
            config_path: Path to the configuration file of the detectron2 model
            opts: Additional option pairs to override settings in the configuration
            post_process: Whether to run post-processing or not
        """
        # Loading and initializing the config
        self.logger = logging.getLogger(__name__)
        self.cfg = self.prepare_cfg(config_path, opts)

        self.model_path = model_path
        self.config_path = config_path
        self.post_process_flag = post_process

        # Logging
        self.logger.info('Building GLASS Text Spotting Model')
        self.logger.info(f'Model path: {model_path}')
        self.logger.info(f'Config path: {config_path}')
        self.logger.info(f'Post-Process: {post_process}')

        # Initializing the architecture
        self.model = build_model(self.cfg)
        self.model.eval()
        self.device = self.model.device

        # Loading the weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_path)

        # Defining the pre-processing transforms (resize, etc...)
        self.min_target_size = self.cfg.INPUT.MIN_SIZE_TEST
        self.max_target_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.max_upscale_ratio = self.cfg.INPUT.MAX_UPSCALE_RATIO
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR", "GREY"], self.input_format

        self.text_encoder = TextEncoder(self.cfg)
        self.post_processor = build_post_processor(self.cfg)

    @staticmethod
    def prepare_cfg( config_path: str, opts: List[str]) -> None:
        cfg = get_cfg()
        add_e2e_config(cfg)
        add_glass_config(cfg)
        add_dataset_config(cfg)
        add_post_process_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.merge_from_list(opts or list())
        return cfg.clone()  # cfg can be modified by model

    def __call__(self, img_list: List[np.ndarray]) -> List[Instances]:
        input_data = self.preprocess(img_list)
        img, im_info = self.convert_batched_inputs_to_c2_format(input_data, 32, self.device)
        with torch.no_grad():
            raw_predictions = self.model(img, im_info)
        preds = self.post_run(raw_predictions, input_data)
        return preds

    def preprocess(self, img_list: List[np.ndarray]) -> List[dict]:
        input_data = []
        for original_image in img_list:
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            if self.input_format == "GREY":
                original_image = rgb2grey(original_image, three_channels=True)
            image_height, image_width = original_image.shape[:2]

            image_tensor, scale_ratio = self._image_to_tensor(original_image, self.device)
            height, width = image_tensor.shape[1:]
            inputs = {'image': image_tensor, 'height': height, 'width': width,
                      'scale_ratio': scale_ratio, 'origin_height': image_height, 'origin_width': image_width}
            input_data.append(inputs)
        return input_data

    @staticmethod
    def convert_batched_inputs_to_c2_format(batched_inputs: list, size_divisibility: int, device: torch.device) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert all(isinstance(x, dict) for x in batched_inputs)
        assert all(x["image"].dim() == 3 for x in batched_inputs)

        images = [x["image"] for x in batched_inputs]
        images = ImageList.from_tensors(images, size_divisibility)
        im_info = torch.Tensor(images.image_sizes).to(device)
        return images.tensor.to(device), im_info
    
    def post_run(self, preds: List[dict], input_data: List[dict]) -> List[Instances]:
        res_per_batch = self.split_preds_to_batches(preds)

        image_shape_list = [(img_data['origin_height'], img_data['origin_width']) for img_data in input_data]
        preds = self.preds2instance(res_per_batch, image_shape_list)
        res = []
        for pred, img_data in zip(preds, input_data):
            scale_ratio = img_data.get('scale_ratio', 1)
            if scale_ratio != 1:
                pred.pred_boxes.scale(1 / scale_ratio, 1 / scale_ratio)
            pred = self.post_processor(pred)
            res.append(pred)
        return res

    def split_preds_to_batches(self, preds):
        pred_boxes, pred_scores, pred_classes, pred_text_prob, results_n_per_batch = preds
        start_idx = 0
        res_per_batch = []
        for res_len in results_n_per_batch:
            res_per_batch.append([pred_boxes[start_idx: start_idx + res_len],
                              pred_scores[start_idx: start_idx + res_len],
                              pred_classes[start_idx: start_idx + res_len],
                              pred_text_prob[start_idx: start_idx + res_len]])
            start_idx += res_len
        return res_per_batch

    @staticmethod
    def preds2instance(preds_list: List, image_shape: List[tuple]) -> List[Instances]:
        """
        Args:
            preds_list: list contains list of the predictions[boxes, scores, classes, text_prob]
            image_shape: list of tuples (height, width)

        Returns: list of Instances
        """
        res_list = []
        for pred, img_size in zip(preds_list, image_shape):
            result = Instances(img_size)
            result.pred_boxes = RotatedBoxes(pred[0])
            result.scores = pred[1]
            result.pred_classes = pred[2]
            result.pred_text_prob = pred[3]
            res_list.append(result)
        return res_list

    def plot_img(self, images: List[np.array], images_names: List[str], out_dir: str, preds: List[Instances],
                 as_html: bool = True) -> None:
        for image, img_name, pred in zip(images, images_names, preds):
            pred = self.post_processor(pred)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            figure = visualize(preds=pred, image=image, text_encoder=self.text_encoder, vis_width=720,  vis_text=True)
            file_path = os.path.join(out_dir, img_name)[:-4]
            figure.write_html(f'{file_path}.html') if as_html else figure.write_image(f'{file_path}.jpg')

    def get_inference_scale_ratio(self, height: int, width: int) -> float:
        max_image_dim = max(height, width)
        if max_image_dim > self.max_target_size:
            scale_ratio = self.max_target_size / max_image_dim
        elif max_image_dim < self.min_target_size:
            scale_ratio = min(self.max_upscale_ratio, self.min_target_size / max_image_dim)
        else:
            scale_ratio = 1
        return scale_ratio

    def _image_to_tensor(self, original_image: np.ndarray, device: torch.device, interpolation: str='bilinear'):
        """
        Transfers the image to the GPU as a resized tensor
        Args:
            original_image: The original image as a numpy array (H, W, C)
            device: The cuda device (or CPU) to which we send the tensor
            interpolation: Either 'nearest' or 'bilinear' are supported for interpolation algorithms
        :return: Both the resized image tensor, and the original image tensor
        """
        height, width = original_image.shape[:2]

        image_tensor = torch.as_tensor(original_image.transpose((2, 0, 1)))
        image_tensor = image_tensor.to(device).to(torch.float32)

        # Computing the necessary scale ratio (> 1 for enlarging image)
        scale_ratio = self.get_inference_scale_ratio(height, width)

        # Resizing if necessary, if not we just clone the image
        if scale_ratio != 1:
            new_height, new_width = int(np.round(scale_ratio * height)), int(np.round(scale_ratio * width))
            image_tensor_resized = torch.nn.functional.interpolate(image_tensor.unsqueeze(dim=0),
                                                                   size=(new_height, new_width),
                                                                   mode=interpolation,
                                                                   align_corners=False).squeeze(dim=0)
        else:
            image_tensor_resized = image_tensor.clone()
        return image_tensor_resized, scale_ratio

    def preds_boxes_to_polygons(self, pred_boxes: torch.tensor) -> torch.tensor:
        box_tensor = pred_boxes.tensor
        polygons = self.post_processor.boxes_to_polygons(boxes=box_tensor)
        return polygons
