"""Configuration objects for vehicle classifier inference and explicit costing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ClassificationConfig:
    """Settings that affect classification inputs and optional post-processing."""

    image_size: tuple[int, int] = (224, 224)  # Input crop size used when resizing before classifier inference.
    pad_fill_rgb: tuple[int, int, int] = (124, 116, 104)  # RGB fill used when padding small crops to a square image.
    normalise_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # Mean values used for input image normalisation.
    normalise_std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # Std dev values used for input image normalisation.
    confidence_threshold: float = 0.0  # Minimum confidence required before a prediction is treated as usable.
    smoothing_window: int = 1  # Number of previous scores averaged before the final class decision.

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain two positive integers")
        if len(self.pad_fill_rgb) != 3 or any(not 0 <= value <= 255 for value in self.pad_fill_rgb):
            raise ValueError("pad_fill_rgb must contain three values in [0, 255]")
        if len(self.normalise_mean) != 3 or len(self.normalise_std) != 3:
            raise ValueError("normalise_mean and normalise_std must contain three values")
        if any(value <= 0 for value in self.normalise_std):
            raise ValueError("normalise_std values must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be at least one")


DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "type": 0.65,
    "car_subtype": 0.20,
    "colour": 0.15,
}


@dataclass(frozen=True)
class ExplicitFeatureConfig:
    """Penalty-matrix paths and weighting used by explicit feature costing."""

    colour_matrix_path: str | Path | None = None  # CSV path for colour-to-colour directional penalty matrix.
    type_matrix_path: str | Path | None = None  # CSV path for vehicle-type directional penalty matrix.
    car_subtype_matrix_path: str | Path | None = None  # CSV path for subtype directional penalty matrix.
    colour_semantic_matrix_path: str | Path | None = None  # CSV path for colour-to-colour semantic penalty matrix.
    type_semantic_matrix_path: str | Path | None = None  # CSV path for vehicle-type semantic penalty matrix.
    car_subtype_semantic_matrix_path: str | Path | None = None  # CSV path for subtype semantic penalty matrix.
    feature_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS)
    )  # Relative importance of each feature when combining explicit costs.
    confidence_weighting: bool = True  # Whether classifier confidence adjusts the feature-cost contribution. Not currently used
    semantic_reliability: bool = True  # Whether semantic reliability should scale disagreement under the matrix term. Not currently used
    semantic_weight: float = 0.6  # Blend between matrix-based semantic cost and divergence-based cost. not used currently 
    missing_cost: float = 1.0  # Fallback cost used when a feature pair cannot be scored.
    adjust_for_car_type: bool = True  # if both confident car, reduce type cost, if not remove subtype, controls if on or off
    explicit_cost_method: str = "explicit_feature_cost_p_norm_combined"  # Name of the ClassifierMetrics method to invoke for explicit feature scoring.
    p_norm_p: float = 3.0  # Power used by the p-norm weighted power-mean aggregation methods (explicit_feature_cost_p_norm*).

    def __post_init__(self) -> None:
        if not self.feature_weights or any(value < 0 for value in self.feature_weights.values()):
            raise ValueError("feature_weights must contain non-negative values")
        if sum(self.feature_weights.values()) <= 0:
            raise ValueError("feature_weights must sum to more than zero")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1]")
        if not 0.0 <= self.missing_cost <= 1.0:
            raise ValueError("missing_cost must be in [0, 1]")
        if self.p_norm_p < 1.0:
            raise ValueError("p_norm_p must be >= 1")
        allowed_methods = {
            "explicit_feature_cost",
            "explicit_feature_cost_p_norm",
            "explicit_feature_cost_p_norm_combined",
        }
        if self.explicit_cost_method not in allowed_methods:
            methods = ", ".join(sorted(allowed_methods))
            raise ValueError(
                f"explicit_cost_method must be one of: {methods}; got {self.explicit_cost_method!r}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Checkpoint, DINOv2, and runtime options for model construction."""

    checkpoint_path: str | Path | None = None  # Path to the classifier checkpoint used to build the inference model.
    dinov2_repo_path: str | Path = "dinov2"  # Local path to the DINOv2 repository used by the backbone.
    device: str | torch.device | None = None  # Device target such as CPU or CUDA for model execution.
    metadata_path: str | Path | None = None  # Optional metadata file used to map classifier outputs to labels.
    backbone_name: str | None = None  # Optional backbone name override used during model construction.
    type_classes: list[str] | None = None  # Label list for vehicle type classes used by the classifier head.

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        values = {
            "checkpoint_path": self.checkpoint_path,
            "dinov2_repo_path": self.dinov2_repo_path,
            "device": self.device,
            "metadata_path": self.metadata_path,
            "backbone_name": self.backbone_name,
            "type_classes": self.type_classes,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return ModelConfig(**values)
