from typing import Optional, Union

from transformers import (
    GroundingDinoConfig,
    GroundingDinoProcessor,
    GroundingDinoForObjectDetection as _GroundingDinoForObjectDetection,
)

from torchvision.transforms.v2.functional import convert_bounding_box_format
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from torchvision.transforms import v2 as T
import torch

from ..base import BaseModel, ModelProvider, WeightsInfo, ObjectDetectionMixin
from ...datasets import BaseDataset, DataPreparation


class GroundingDinoDataPreparation(DataPreparation):
    """Generic data preparation for Grounding DINO.

    Two operating modes are supported:

    * Closed-set / supervised: pass a fixed ``text_labels`` list at construction
      time. The same list of phrases is fed to ``GroundingDinoProcessor`` for
      every sample in the batch, so no per-sample text prompt is required.
    * Zero-shot: leave ``text_labels=None``. Each sample's ``target`` dict must
      then carry a list of candidate phrases under
      ``dataset_key['text_labels']``.
    """

    model_id = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = dict(
            bboxes="boxes2d",
            classes="boxes2d_classes",
            original_size="original_hw",
            text_labels="text_labels",
        ),
        img_size: int = 800,
        longest_edge: int = 1333,
        evaluation_mode: bool = True,
        confidence_threshold: float = 0.25,
        text_threshold: float = 0.25,
        text_labels: Optional[list[str]] = None,
        text_template: str = "{}",
        default_augment: T.Compose = T.Compose([
            T.RandomHorizontalFlip()
        ]),
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.longest_edge = longest_edge
        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold
        self.evaluation_mode = evaluation_mode
        self.text_template = text_template

        self.text_labels = (
            None if text_labels is None
            else [text_template.format(label) for label in text_labels]
        )

        if evaluation_mode:
            self.default_augment = lambda inputs: inputs
        else:
            self.default_augment = default_augment

        self.processor = GroundingDinoProcessor.from_pretrained(self.model_id)
        self.processor.image_processor.size = {"shortest_edge": self.img_size, "longest_edge": self.longest_edge}
        self.processor.image_processor.do_resize = True

        self._last_text_per_sample: Optional[list[list[str]]] = None

    def transforms(self, *args, idx=None):
        image, target = args[0] if len(args) == 1 else args

        bbox = target.get(self.dataset_key['bboxes']) if isinstance(target, dict) else None
        if bbox is not None:
            bbox_classes = target[self.dataset_key['classes']]
            img_size = target[self.dataset_key['original_size']]

            image, bbox = self.default_augment((image, bbox))

            if not isinstance(bbox, BoundingBoxes):
                bbox = BoundingBoxes(bbox, format=BoundingBoxFormat.XYXY, canvas_size=img_size)
            if bbox.format != BoundingBoxFormat.XYWH:
                bbox = convert_bounding_box_format(bbox, new_format=BoundingBoxFormat.XYWH)

            annotations = []
            new_target = dict(image_id=idx, annotations=annotations)
            if self.text_labels is None and self.dataset_key['text_labels'] in target:
                new_target[self.dataset_key['text_labels']] = target[self.dataset_key['text_labels']]
            for box, cls in zip(bbox, bbox_classes):
                width, height = box[2:].tolist()
                annotations.append(dict(
                    bbox=box,
                    category_id=cls.item(),
                    area=width * height,
                    iscrowd=0,
                ))
            target = new_target

        if len(args) == 1:
            return dict(image=image, target=target)
        else:
            return image, target

    def _resolve_text(self, targets, batch_size):
        if self.text_labels is not None:
            return [self.text_labels] * batch_size

        per_sample = []
        for target in targets:
            labels = target.get(self.dataset_key['text_labels']) if isinstance(target, dict) else None
            if labels is None:
                raise ValueError(
                    f"Zero-shot Grounding DINO requires per-sample candidate phrases under "
                    f"target['{self.dataset_key['text_labels']}'], or a fixed `text_labels` "
                    f"argument on GroundingDinoDataPreparation."
                )
            per_sample.append([self.text_template.format(label) for label in labels])
        return per_sample

    def pre_process(self, batch):
        images, targets = batch
        text = self._resolve_text(targets, len(images))
        self._last_text_per_sample = text

        none_idx_found = False
        annotations: Optional[list] = []
        for target in targets:
            if isinstance(target, dict) and target.get('annotations') is not None:
                for annotation in target['annotations']:
                    if isinstance(annotation.get('bbox'), torch.Tensor):
                        annotation['bbox'] = annotation['bbox'].tolist()
                annotations.append(target)
                if not target.get('image_id', None):
                    none_idx_found = True
            else:
                annotations = None  # mixed / inference-only batch — skip annotations entirely
                break

        if none_idx_found:
            for i, target in enumerate(annotations):
                target['image_id'] = i

        if annotations:
            return self.processor(
                images=images, text=text, annotations=annotations, return_tensors="pt"
            )
        return self.processor(images=images, text=text, return_tensors="pt")

    def post_process(self, batch, target_sizes=None, input_ids=None):
        results = self.processor.post_process_grounded_object_detection(
            batch,
            input_ids=input_ids,
            threshold=self.confidence_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        # `post_process_grounded_object_detection` returns the matched text spans
        # under both `labels` and `text_labels`. Downstream consumers
        # (e.g. supervision.Detections.from_transformers, mAP eval)
        # expect `labels` to be a tensor of integer class IDs aligned with the
        # candidate phrase order. Map text -> index here.
        candidates_per_sample = self._candidates_per_sample(len(results))
        device = batch.logits.device if hasattr(batch, "logits") else torch.device("cpu")
        for result, candidates in zip(results, candidates_per_sample):
            text_labels = result.get("text_labels")
            if text_labels is None:
                text_labels = result.get("labels", [])
            class_ids = [self._match_label_index(t, candidates) for t in text_labels]
            result["labels"] = torch.tensor(class_ids, dtype=torch.long, device=device)
            result["text_labels"] = list(text_labels)
        return results

    def _candidates_per_sample(self, batch_size: int) -> list[list[str]]:
        if self.text_labels is not None:
            return [self.text_labels] * batch_size
        if self._last_text_per_sample is not None and len(self._last_text_per_sample) == batch_size:
            return self._last_text_per_sample
        return [[] for _ in range(batch_size)]

    @staticmethod
    def _match_label_index(text: str, candidates: list[str]) -> int:
        if not candidates:
            return -1
        norm = text.strip().lower()
        normalized_candidates = [c.strip().lower() for c in candidates]
        for i, cand in enumerate(normalized_candidates):
            if norm == cand:
                return i
        for i, cand in enumerate(normalized_candidates):
            if norm and (norm in cand or cand in norm):
                return i
        # token-overlap fallback for partial-phrase matches
        best_i, best_overlap = -1, 0
        norm_tokens = set(norm.split())
        for i, cand in enumerate(normalized_candidates):
            overlap = len(norm_tokens & set(cand.split()))
            if overlap > best_overlap:
                best_overlap, best_i = overlap, i
        return best_i

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx], idx=idx)

    def collate_fn(self, batch):
        try:
            images = [item['image'] for item in batch]
            targets = [item['target'] for item in batch]
        except TypeError:
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
        return self.pre_process((images, targets))


class GroundingDinoFixedClassDataPreparation(GroundingDinoDataPreparation):
    """DataPreparation that auto-injects ``dataset.classes`` as the fixed text prompt.

    Used by :class:`GroundingDinoForObjectDetection` so the user never has to
    supply text prompts when the candidate label set is fixed by the dataset.
    """

    def __init__(self, dataset: BaseDataset, **kwargs):
        kwargs.setdefault("text_labels", list(dataset.classes))
        super().__init__(dataset, **kwargs)


class GroundingDinoForZeroShotObjectDetection(BaseModel, _GroundingDinoForObjectDetection, ObjectDetectionMixin):
    """Open-vocabulary Grounding DINO. Text prompts are supplied at runtime."""

    model_id = "IDEA-Research/grounding-dino-tiny"
    model_name = "GroundingDINO-Tiny"
    model_provider = ModelProvider.HuggingFace
    DataPreparation = GroundingDinoDataPreparation

    class ModelRegistry:
        TINY_OFFICIAL = WeightsInfo("IDEA-Research/grounding-dino-tiny")
        BASE_OFFICIAL = WeightsInfo("IDEA-Research/grounding-dino-base")

    def __init__(
        self,
        config: Optional[GroundingDinoConfig] = None,
        dataset: Union[BaseDataset, str] = "",
        **kwargs,
    ):
        if config is None:
            config = GroundingDinoConfig.from_pretrained(self.model_id)
        super(BaseModel, self).__init__(config=config, **kwargs)  # skip BaseModel.__init__

        self.dataset_name = dataset if isinstance(dataset, str) else dataset.dataset_name
        self.num_classes = len(dataset.classes) if isinstance(dataset, BaseDataset) else 0

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):  # NOTE: Method kwargs must be explicitly listed for Trainer to work properly
        return super(BaseModel, self).forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class GroundingDinoForObjectDetection(GroundingDinoForZeroShotObjectDetection):
    """Closed-set Grounding DINO bound to a fixed dataset class list.

    Identical model architecture to :class:`GroundingDinoForZeroShotObjectDetection`;
    the only difference is that its :class:`DataPreparation` automatically uses
    ``dataset.classes`` as the text prompt for every sample, so callers do not
    need to manage text prompts manually.
    """

    model_name = "GroundingDINO-OD-Tiny"
    DataPreparation = GroundingDinoFixedClassDataPreparation


GroundingDinoForZeroShotObjectDetection.ModelRegistry.TINY = GroundingDinoForZeroShotObjectDetection.ModelRegistry.TINY_OFFICIAL
GroundingDinoForZeroShotObjectDetection.ModelRegistry.BASE = GroundingDinoForZeroShotObjectDetection.ModelRegistry.BASE_OFFICIAL

# Grounding DINO is open-vocabulary; the same pretrained weights work across datasets,
# so every dataset alias just points at the canonical TINY checkpoint.
for _alias in ("COCO", "SHIFT", "SHIFT_SUBSET", "CityScapes", "ACDC"):
    setattr(
        GroundingDinoForZeroShotObjectDetection.ModelRegistry,
        _alias,
        GroundingDinoForZeroShotObjectDetection.ModelRegistry.TINY_OFFICIAL,
    )
del _alias
