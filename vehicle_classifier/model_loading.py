"""Checkpoint metadata, architecture detection, and runtime model loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch import nn

from .config import ModelConfig
from .modeling import build_model_for_checkpoint


class LoadedClassifier:
    """Resolved model and metadata produced by checkpoint loading."""

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        device: torch.device,
        colour_classes: list[str],
        type_classes: list[str],
        car_subtype_classes: list[str],
    ) -> None:
        """Store the loaded runtime model and its checkpoint metadata."""
        self.model = model
        self.variant = variant
        self.device = device
        self.colour_classes = colour_classes
        self.type_classes = type_classes
        self.car_subtype_classes = car_subtype_classes


def load_classifier(
    config: ModelConfig,
    *,
    expected_variant: str | None = None,
    type_classes_override: list[str] | None = None,
) -> LoadedClassifier:
    """Load a checkpoint, infer its variant, and return the ready-to-use model bundle."""
    if config.checkpoint_path is None:
        raise ValueError("ModelConfig.checkpoint_path is required")
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_any = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_any, Mapping):
        raise TypeError("Checkpoint must be a mapping/object serialized by torch.save")
    checkpoint = cast(Mapping[str, Any], checkpoint_any)
    metadata = _load_metadata(config.metadata_path)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        checkpoint_config = metadata.get("config") if isinstance(metadata.get("config"), Mapping) else {}
    backbone_name = config.backbone_name or checkpoint_config.get("backbone_name")
    if not isinstance(backbone_name, str) or not backbone_name:
        raise ValueError(
            "Missing backbone name for checkpoint loading. Provide 'backbone_name' "
            "or pass metadata_path with JSON containing config.backbone_name."
        )

    colour_classes = _class_list(checkpoint.get("colour_classes", metadata.get("colour_classes", [])))
    type_classes_value = checkpoint.get(
        "type_classes", metadata.get("type_classes", type_classes_override or config.type_classes)
    )
    type_classes = _class_list(type_classes_value)
    if not type_classes:
        type_classes = [str(index) for index in range(_infer_type_class_count(state_dict))]
    car_subtype_classes = _class_list(
        checkpoint.get("car_subtype_classes", metadata.get("car_subtype_classes", []))
    )

    model, variant = build_model_for_checkpoint(
        state_dict,
        backbone_name=backbone_name,
        n_colours=len(colour_classes),
        n_types=len(type_classes),
        n_subtypes=len(car_subtype_classes),
        dinov2_repo_path=config.dinov2_repo_path,
    )
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"Checkpoint variant mismatch: expected {expected_variant!r}, detected {variant!r}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return LoadedClassifier(
        model,
        variant,
        device,
        colour_classes,
        type_classes,
        car_subtype_classes,
    )


def _class_list(value: Any) -> list[str]:
    """Return a class list only when the checkpoint metadata exposes one."""
    return list(value) if isinstance(value, list) else []


def _load_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    """Load a JSON metadata file that supplements checkpoint labels and config."""
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
    """Infer the number of type labels from the checkpoint head shape when unavailable."""
    for key in ("type_head.3.weight", "type_head.3.bias"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError(
        "Could not infer number of type classes from checkpoint state_dict. "
        "Provide metadata_path with 'type_classes'."
    )
