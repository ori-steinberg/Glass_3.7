import torch
from typing import List, Optional, Dict
from detectron2.config import configurable
from detectron2.structures import Instances
from glass.postprocess import build_post_processor
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from glass.postprocess.post_processor_academic import detector_postprocess


@META_ARCH_REGISTRY.register()
class GlassRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(self, *, post_processor, inflate_ratio, filter_small_boxes, transcript_filtering,
                 drop_overlapping_boxes,
                 ioa_threshold, valid_score, **kwargs):
        super().__init__(**kwargs)
        self.post_processor = post_processor
        self.inflate_ratio = inflate_ratio
        self.transcript_filtering = transcript_filtering
        self.filter_small_boxes = filter_small_boxes
        self.ioa_threshold = ioa_threshold
        self.valid_score = valid_score
        self.drop_overlapping_boxes = drop_overlapping_boxes

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['post_processor'] = build_post_processor(cfg)

        ret['inflate_ratio'] = cfg.POST_PROCESSING.INFLATE_RATIO if hasattr(cfg.POST_PROCESSING, 'INFLATE_RATIO') \
            else None
        ret['transcript_filtering'] = cfg.POST_PROCESSING.TRANSCRIPT_FILTERING if hasattr(cfg.POST_PROCESSING,
                                                                                          'TRANSCRIPT_FILTERING') \
            else None
        ret['filter_small_boxes'] = cfg.POST_PROCESSING.MIN_BOX_DIMENSION if hasattr(cfg.POST_PROCESSING,
                                                                                     'MIN_BOX_DIMENSION') \
            else None
        ret['drop_overlapping_boxes'] = cfg.POST_PROCESSING.DROP_OVERLAPPING if hasattr(cfg.POST_PROCESSING,
                                                                                        'DROP_OVERLAPPING') \
            else None
        ret['ioa_threshold'] = cfg.POST_PROCESSING.IOA_THRESHOLD if hasattr(cfg.POST_PROCESSING, 'IOA_THRESHOLD') \
            else None
        ret['valid_score'] = cfg.INFERENCE_TH_TEST if hasattr(cfg, 'INFERENCE_TH_TEST') else 0

        return ret

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, pred_text_prob = self.roi_heads(images.tensor, features, proposals, None)

        results = [pred.get_fields() for pred in results]
        for res in results:
            res['pred_boxes'] = res['pred_boxes'].tensor
        return tuple(results), pred_text_prob

    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []

        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            r = detector_postprocess(results_per_image, input_per_image)

            processed_results.append({"instances": r})
        return processed_results
