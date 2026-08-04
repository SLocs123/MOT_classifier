from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import cast

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .modeling import build_model_for_checkpoint
from .modeling import variant_uses_image_size


IMAGENET_MEAN_RGB_255 = (124, 116, 104)


def encode_image_size(image_width: int, image_height: int) -> torch.Tensor:
    area = image_width * image_height
    aspect = image_width / max(image_height, 1)
    return torch.tensor(
        [
            math.log(area + 1.0),
            math.log(aspect + 1e-6),
        ],
        dtype=torch.float32,
    )


class ResizeLongestSideAndPad:
    """Resize while preserving aspect ratio, then pad to a fixed canvas."""

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        fill: tuple[int, int, int] = IMAGENET_MEAN_RGB_255,
    ):
        self.target_size = target_size
        self.fill = fill
        if hasattr(Image, "Resampling"):
            self.resample = Image.Resampling.BICUBIC
        else:
            self.resample = 3

    def __call__(self, image: Image.Image) -> Image.Image:
        target_width, target_height = self.target_size
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Image must have positive width and height")

        scale = min(target_width / width, target_height / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))

        resized = image.resize((resized_width, resized_height), resample=self.resample)
        canvas = Image.new(image.mode, (target_width, target_height), self.fill)

        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


DEFAULT_EVAL_TRANSFORM = transforms.Compose(
    [
        ResizeLongestSideAndPad(
            target_size=(224, 224),
            fill=IMAGENET_MEAN_RGB_255,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class VehicleClassifier:
    """Inference wrapper for vehicle attribute checkpoints from this repository."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        expected_variant: str | None = None,
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt_any = torch.load(checkpoint_path, map_location="cpu")
        ckpt: Mapping[str, Any]
        if not isinstance(ckpt_any, Mapping):
            raise TypeError("Checkpoint must be a mapping/object serialized by torch.save")
        ckpt = cast(Mapping[str, Any], ckpt_any)

        metadata = _load_metadata(metadata_path)

        if "model" in ckpt and isinstance(ckpt["model"], Mapping):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        cfg_any = ckpt.get("config")
        if not isinstance(cfg_any, Mapping):
            cfg_any = metadata.get("config") if isinstance(metadata.get("config"), Mapping) else {}
        config = cast(Mapping[str, Any], cfg_any)

        resolved_backbone_name = backbone_name or config.get("backbone_name")
        if not isinstance(resolved_backbone_name, str) or not resolved_backbone_name:
            raise ValueError(
                "Missing backbone name for checkpoint loading. Provide 'backbone_name' "
                "or pass metadata_path with JSON containing config.backbone_name."
            )

        ckpt_colour_classes = ckpt.get("colour_classes", metadata.get("colour_classes", []))
        ckpt_type_classes = ckpt.get("type_classes", metadata.get("type_classes", type_classes))
        ckpt_car_subtype_classes = ckpt.get("car_subtype_classes", metadata.get("car_subtype_classes", []))

        self.colour_classes = list(ckpt_colour_classes) if isinstance(ckpt_colour_classes, list) else []
        self.car_subtype_classes = list(ckpt_car_subtype_classes) if isinstance(ckpt_car_subtype_classes, list) else []

        if isinstance(ckpt_type_classes, list) and len(ckpt_type_classes) > 0:
            self.type_classes = list(ckpt_type_classes)
        else:
            n_types = _infer_type_class_count(state_dict)
            self.type_classes = [str(i) for i in range(n_types)]

        self.model, self.variant = build_model_for_checkpoint(
            state_dict,
            backbone_name=resolved_backbone_name,
            n_colours=len(self.colour_classes),
            n_types=len(self.type_classes),
            n_subtypes=len(self.car_subtype_classes),
            dinov2_repo_path=dinov2_repo_path,
        )
        if expected_variant is not None and self.variant != expected_variant:
            raise ValueError(
                f"Checkpoint variant mismatch: expected {expected_variant!r}, detected {self.variant!r}"
            )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = DEFAULT_EVAL_TRANSFORM

    def _prepare_image(self, image: str | Path | Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not image_path.is_file():
                raise IsADirectoryError(f"Image path must be a file: {image_path}")
            pil_img = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError("image must be a path or PIL.Image.Image")

        image_width, image_height = pil_img.size
        transformed = self.transform(pil_img)
        tensor = cast(torch.Tensor, transformed).unsqueeze(0)
        image_size = encode_image_size(image_width, image_height).unsqueeze(0)
        return tensor.to(self.device), image_size.to(self.device)

    def _forward(self, image_batch: torch.Tensor, image_size_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if variant_uses_image_size(self.variant):
            return self.model(image_batch, image_size_batch)
        return self.model(image_batch)

    @torch.no_grad()
    def results(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Return top-1 label, index, and confidence for each head."""
        batch, image_size = self._prepare_image(image)
        outputs = self._forward(batch, image_size)

        result: dict[str, Any] = {}

        if "colour" in outputs:
            colour_probs = F.softmax(outputs["colour"], dim=1)[0]
            colour_idx = int(torch.argmax(colour_probs).item())
            result["colour"] = {
                "index": colour_idx,
                "label": self.colour_classes[colour_idx],
                "confidence": float(colour_probs[colour_idx].item()),
                "probs": colour_probs.cpu().tolist(),
            }

        if "type" in outputs:
            type_probs = F.softmax(outputs["type"], dim=1)[0]
            type_idx = int(torch.argmax(type_probs).item())
            result["type"] = {
                "index": type_idx,
                "label": self.type_classes[type_idx],
                "confidence": float(type_probs[type_idx].item()),
                "probs": type_probs.cpu().tolist(),
            }
        else:
            raise ValueError("Loaded checkpoint model did not produce a type prediction")

        if "car_subtype" in outputs:
            subtype_probs = F.softmax(outputs["car_subtype"], dim=1)[0]
            subtype_idx = int(torch.argmax(subtype_probs).item())
            result["car_subtype"] = {
                "index": subtype_idx,
                "label": self.car_subtype_classes[subtype_idx],
                "confidence": float(subtype_probs[subtype_idx].item()),
                "probs": subtype_probs.cpu().tolist(),
            }

        result["checkpoint_variant"] = self.variant
        return result


class VehicleClassifierBase(VehicleClassifier):
    """Inference wrapper for checkpoints from train_classifiers.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            expected_variant="baseline",
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )


class VehicleClassifierWithSize(VehicleClassifier):
    """Inference wrapper for checkpoints from train_classifiers_with_size.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            expected_variant="with_size",
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )


class VehicleClassifierIntermediate(VehicleClassifier):
    """Inference wrapper for checkpoints from train_classifiers_intermediate.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            expected_variant="intermediate",
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )


class VehicleClassifierIntermediateWithSize(VehicleClassifier):
    """Inference wrapper for checkpoints from train_classifiers_intermediate_with_size.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            expected_variant="intermediate_with_size",
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )


class VehicleClassifierTypeOnly(VehicleClassifier):
    """Inference wrapper for type-only checkpoints from type-only training runs."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            expected_variant=None,
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )
        if not self.variant.startswith("type_only"):
            raise ValueError(
                f"Checkpoint does not appear to be a type-only checkpoint: detected variant {self.variant!r}"
            )


def _load_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}

    path = Path(metadata_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Metadata file is not valid JSON: {path}") from error

    if not isinstance(data, dict):
        raise ValueError("Metadata JSON must be an object/dict")
    return data


def _infer_type_class_count(state_dict: Mapping[str, Any]) -> int:
    candidate_keys = [
        "type_head.3.weight",
        "type_head.3.bias",
        "type_head.0.weight",
    ]
    for key in candidate_keys:
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            if key.endswith(".0.weight"):
                # First layer weight shape: [hidden_dim, in_dim]
                continue
            return int(value.shape[0])

    raise ValueError(
        "Could not infer number of type classes from checkpoint state_dict. "
        "Provide metadata_path with 'type_classes'."
    )


def vehicle_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifier:
    """Factory function requested API: vehicle_classifier('xxx.pt')."""
    return VehicleClassifier(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )


def vehicle_classifier_base(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierBase:
    return VehicleClassifierBase(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )


def vehicle_classifier_with_size(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierWithSize:
    return VehicleClassifierWithSize(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )


def vehicle_classifier_intermediate(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierIntermediate:
    return VehicleClassifierIntermediate(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )


def vehicle_classifier_intermediate_with_size(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierIntermediateWithSize:
    return VehicleClassifierIntermediateWithSize(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )


def vehicle_classifier_type_only(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierTypeOnly:
    return VehicleClassifierTypeOnly(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )