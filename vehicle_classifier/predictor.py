from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import ClassificationConfig
from .config import ModelConfig
from .model_loading import load_classifier
from .modeling import variant_uses_image_size


IMAGENET_MEAN_RGB_255 = (124, 116, 104)


def encode_image_size(image_width: int, image_height: int) -> torch.Tensor:
    """Convert image dimensions into the scalar size feature used by size-aware variants."""
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


def build_eval_transform(config: ClassificationConfig) -> transforms.Compose:
    """Create the deterministic preprocess pipeline used before model inference."""
    return transforms.Compose(
        [
            ResizeLongestSideAndPad(
                target_size=config.image_size,
                fill=config.pad_fill_rgb,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalise_mean, std=config.normalise_std),
        ]
    )


DEFAULT_EVAL_TRANSFORM = build_eval_transform(ClassificationConfig())


class VehicleClassifier:
    """Inference wrapper for vehicle attribute checkpoints from this repository."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path | None = None,
        expected_variant: str | None = None,
        metadata_path: str | Path | None = None,
        backbone_name: str | None = None,
        type_classes: list[str] | None = None,
        model_config: ModelConfig | None = None,
        classification_config: ClassificationConfig | None = None,
    ):
        """Load the correct checkpoint variant and attach the image preprocessing pipeline."""
        resolved_model_config = (model_config or ModelConfig()).with_overrides(
            checkpoint_path=checkpoint_path,
            device=device,
            dinov2_repo_path=dinov2_repo_path,
            metadata_path=metadata_path,
            backbone_name=backbone_name,
            type_classes=type_classes,
        )
        loaded = load_classifier(
            resolved_model_config,
            expected_variant=expected_variant,
            type_classes_override=type_classes,
        )
        self.device = loaded.device
        self.model = loaded.model
        self.variant = loaded.variant
        self.colour_classes = loaded.colour_classes
        self.type_classes = loaded.type_classes
        self.car_subtype_classes = loaded.car_subtype_classes
        self.classification_config = classification_config or ClassificationConfig()
        self.transform = build_eval_transform(self.classification_config)

    def _prepare_image(self, image: str | Path | Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an image and convert it to the tensor layout expected by the model."""
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
        """Run the model with the correct signature for the checkpoint variant being used."""
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

        result["feature_vector"] = outputs["backbone_features"][0].cpu().tolist()
        result["checkpoint_variant"] = self.variant
        return result


class VehicleClassifierBase(VehicleClassifier):
    """Inference wrapper for checkpoints from train_classifiers.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path | None = None,
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
        dinov2_repo_path: str | Path | None = None,
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
        dinov2_repo_path: str | Path | None = None,
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
        dinov2_repo_path: str | Path | None = None,
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
        dinov2_repo_path: str | Path | None = None,
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


def vehicle_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
    model_config: ModelConfig | None = None,
    classification_config: ClassificationConfig | None = None,
) -> VehicleClassifier:
    """Factory entry point for the default vehicle classifier inference wrapper."""
    return VehicleClassifier(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
        model_config=model_config,
        classification_config=classification_config,
    )


def vehicle_classifier_base(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierBase:
    """Construct a baseline checkpoint wrapper for standard model variants."""
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
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierWithSize:
    """Construct a size-aware checkpoint wrapper that includes image-size conditioning."""
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
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierIntermediate:
    """Construct a wrapper for intermediate-layer checkpoint variants used for richer embeddings."""
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
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierIntermediateWithSize:
    """Construct the intermediate size-aware wrapper for checkpoints that use both block aggregation and size conditioning."""
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
    dinov2_repo_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    backbone_name: str | None = None,
    type_classes: list[str] | None = None,
) -> VehicleClassifierTypeOnly:
    """Construct a type-only checkpoint wrapper used when colour and subtype heads are unavailable."""
    return VehicleClassifierTypeOnly(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
        metadata_path=metadata_path,
        backbone_name=backbone_name,
        type_classes=type_classes,
    )