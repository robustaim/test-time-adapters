from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling import GeneralizedRCNN, SwinTransformer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo

from torch import hub, nn

import warnings

from ..base import BaseModel
from ...datasets import BaseDataset


class FasterRCNNForObjectDetection(GeneralizedRCNN, BaseModel):
    model_name = "Faster_RCNN-R50"
    model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    class Weights:
        IMAGENET = lambda: hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl", map_location="cpu")
        NATUREYOO = lambda: hub.load_state_dict_from_url("https://github.com/robustaim/ContinualTTA_ObjectDetection/releases/download/backbone/Faster_R-CNN_Resnet_50_SHIFT.pth", map_location="cpu")

    def __init__(self, dataset: BaseDataset):
        num_classes = len(dataset.classes)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*To copy construct from a tensor.*')
            modules = build_model(cfg)
            super().__init__(
                backbone=modules.backbone,
                proposal_generator=modules.proposal_generator,
                roi_heads=modules.roi_heads,
                pixel_mean=modules.pixel_mean,
                pixel_std=modules.pixel_std,
                input_format=modules.input_format,
                vis_period=modules.vis_period,
            )

        self.dataset_name = dataset.dataset_name
        self.num_classes = num_classes


class SwinRCNNForObjectDetection(GeneralizedRCNN, BaseModel):
    model_name = "SwinT_RCNN-Tiny"
    default_params = dict(
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        frozen_stages=-1,
        out_indices=(0, 1, 2, 3)
    )

    class Weights:
        NATUREYOO = lambda: hub.load_state_dict_from_url("https://github.com/robustaim/ContinualTTA_ObjectDetection/releases/download/backbone/Faster_R-CNN_SwinT_Tiny_SHIFT.pth", map_location="cpu")

    def __init__(self, dataset: BaseDataset):
        num_classes = len(dataset.classes)

        cfg = get_cfg()
        base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(base_config))

        cfg.MODEL.MASK_ON = False
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
        cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            modules = build_model(cfg)
            cfg.MODEL.FPN.IN_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

            swin_backbone = SwinTransformer(**self.default_params)
            swin_backbone._out_features = ["stage{}".format(i+2) for i in swin_backbone.out_indices]
            swin_backbone._out_feature_channels = {
                "stage{}".format(i+2): swin_backbone.embed_dim * 2**i
                for i in swin_backbone.out_indices
            }
            swin_backbone._out_feature_strides = {
                "stage{}".format(i+2): 2 ** (i + 2)
                for i in swin_backbone.out_indices
            }
            original_forward = swin_backbone.forward

            def patched_forward(x):
                outs_orig = original_forward(x)
                outs = {}
                for i in swin_backbone.out_indices:
                    outs["stage{}".format(i+2)] = outs_orig["p{}".format(i)]
                return outs

            swin_backbone.forward = patched_forward

            super().__init__(
                backbone=FPN(
                    bottom_up=swin_backbone,
                    in_features=cfg.MODEL.FPN.IN_FEATURES,
                    out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
                    norm=cfg.MODEL.FPN.NORM,
                    top_block=LastLevelMaxPool(),
                    fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
                ),
                proposal_generator=modules.proposal_generator,
                roi_heads=modules.roi_heads,
                pixel_mean=modules.pixel_mean,
                pixel_std=modules.pixel_std,
                input_format=modules.input_format,
                vis_period=modules.vis_period,
            )

        self.dataset_name = dataset.dataset_name
        self.num_classes = num_classes
