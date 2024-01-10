import mock
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from typing import Dict, List, Optional, Tuple
import torch
from detectron2.structures import ImageList

def GeneralizedRCNN__preprocess_image(self, batched_inputs: torch.Tensor, im_info: torch.Tensor):
    """
    Normalize, pad and batch the input images.
    """
    # images = self._move_to_current_device(batched_inputs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = batched_inputs.to(device)
    normalized_data = (images - self.pixel_mean) / self.pixel_std
    images = ImageList(tensor=normalized_data, image_sizes=im_info)
    return images

patches = (
    mock.patch.object(GeneralizedRCNN, 'preprocess_image', GeneralizedRCNN__preprocess_image),
)