"""Image cropping and explicit vehicle-attribute feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .config import ClassificationConfig
from .config import ExplicitFeatureConfig
from .metrics import ClassifierMetrics
from .config import ModelConfig
from .predictor import vehicle_classifier_intermediate

# TODO: add a method to get the feature embedding from the dinov2 backbone itself, so dont need fast reid and have a complete end-to-end solution for vehicle classification and tracking, will probably need to go in modelling
# TODO: update the cost functioning so that it is correctly normalised to give a reasonable cost between 0 and 1. It is currently too low
# TODO: update/rechech all semantic matrices, ensure that colour is correctly semantically alligned, type should be fairly good, subtype needs to be scaled, lower costs to type, to represent the more difficult classification task.
class ExplicitFeatureExtractor:
    """Builds classifier embeddings and scores track/detection comparisons using explicit feature costs."""

    def __init__(
        self,
        checkpoints_path: str | Path | None = None,
        dinov2_path: str | Path | None = None,
        classifier: Any | None = None,
        classifier_metrics: ClassifierMetrics | None = None,
        *,
        model_config: ModelConfig | None = None,
        classification_config: ClassificationConfig | None = None,
        explicit_config: ExplicitFeatureConfig | None = None,
    ) -> None:
        """Prepare the model, image settings, and explicit cost configuration for later scoring."""
        self.model_config = (model_config or ModelConfig()).with_overrides(
            checkpoint_path=checkpoints_path,
            dinov2_repo_path=dinov2_path,
        )
        self.classification_config = classification_config or ClassificationConfig()
        self.explicit_config = explicit_config or ExplicitFeatureConfig()

        if classifier is not None:
            self.model = classifier
        else:
            if self.model_config.checkpoint_path is None:
                raise ValueError("checkpoints_path is required when classifier is not provided")
            self.model = vehicle_classifier_intermediate(
                checkpoint_path=self.model_config.checkpoint_path,
                device=self.model_config.device,
                dinov2_repo_path=self.model_config.dinov2_repo_path,
                metadata_path=self.model_config.metadata_path,
                backbone_name=self.model_config.backbone_name,
                type_classes=self.model_config.type_classes,
            )
        self.classifier_metrics = classifier_metrics

    def get_classifier_model(self) -> Any:
        """Return the configured inference model used to generate classifier embeddings."""
        return self.model

    def get_colour_classes(self) -> list[str]:
        """Return the colour-label list used by the underlying classifier head."""
        return list(self.model.colour_classes)

    def get_type_classes(self) -> list[str]:
        """Return the vehicle-type label list used by the underlying classifier head."""
        return list(self.model.type_classes)

    def get_car_subtype_classes(self) -> list[str]:
        """Return the subtype label list used by the underlying classifier head."""
        return list(self.model.car_subtype_classes)

    def configure_cost_from_csv(
        self,
        *,
        explicit_config: ExplicitFeatureConfig | None = None,
        colour_matrix_path: str | Path | None = None,
        type_matrix_path: str | Path | None = None,
        car_subtype_matrix_path: str | Path | None = None,
        colour_semantic_matrix_path: str | Path | None = None,
        type_semantic_matrix_path: str | Path | None = None,
        car_subtype_semantic_matrix_path: str | Path | None = None,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool | None = None,
        semantic_reliability: bool | None = None,
        semantic_weight: float | None = None,
        adjust_for_car_type: bool | None = None,
        explicit_cost_method: str | None = None,
        p_norm_p: float | None = None,
    ) -> ClassifierMetrics:
        """Build a cost model from matrix CSV files or a full ExplicitFeatureConfig object."""
        if explicit_config is not None:
            config = explicit_config
        else:
            config = ExplicitFeatureConfig(
                colour_matrix_path=colour_matrix_path or self.explicit_config.colour_matrix_path,
                type_matrix_path=type_matrix_path or self.explicit_config.type_matrix_path,
                car_subtype_matrix_path=car_subtype_matrix_path or self.explicit_config.car_subtype_matrix_path,
                colour_semantic_matrix_path=(
                    colour_semantic_matrix_path
                    if colour_semantic_matrix_path is not None
                    else self.explicit_config.colour_semantic_matrix_path
                ),
                type_semantic_matrix_path=(
                    type_semantic_matrix_path
                    if type_semantic_matrix_path is not None
                    else self.explicit_config.type_semantic_matrix_path
                ),
                car_subtype_semantic_matrix_path=(
                    car_subtype_semantic_matrix_path
                    if car_subtype_semantic_matrix_path is not None
                    else self.explicit_config.car_subtype_semantic_matrix_path
                ),
                feature_weights=self.explicit_config.feature_weights if feature_weights is None else feature_weights,
                confidence_weighting=(
                    self.explicit_config.confidence_weighting
                    if confidence_weighting is None
                    else confidence_weighting
                ),
                semantic_reliability=(
                    self.explicit_config.semantic_reliability
                    if semantic_reliability is None
                    else semantic_reliability
                ),
                semantic_weight=(
                    self.explicit_config.semantic_weight if semantic_weight is None else semantic_weight
                ),
                missing_cost=self.explicit_config.missing_cost,
                adjust_for_car_type=(
                    self.explicit_config.adjust_for_car_type
                    if adjust_for_car_type is None
                    else adjust_for_car_type
                ),
                explicit_cost_method=(
                    self.explicit_config.explicit_cost_method
                    if explicit_cost_method is None
                    else explicit_cost_method
                ),
                p_norm_p=self.explicit_config.p_norm_p if p_norm_p is None else p_norm_p,
            )
        required_paths = {
            "colour_matrix_path": config.colour_matrix_path,
            "type_matrix_path": config.type_matrix_path,
            "car_subtype_matrix_path": config.car_subtype_matrix_path,
        }
        if any(path is None for path in required_paths.values()):
            missing = [name for name, path in required_paths.items() if path is None]
            raise ValueError(f"Missing explicit cost configuration: {', '.join(missing)}")

        colour_matrix_path = config.colour_matrix_path
        type_matrix_path = config.type_matrix_path
        car_subtype_matrix_path = config.car_subtype_matrix_path
        assert colour_matrix_path is not None
        assert type_matrix_path is not None
        assert car_subtype_matrix_path is not None

        metrics = ClassifierMetrics.from_csv_files(
            colour_matrix_path=colour_matrix_path,
            type_matrix_path=type_matrix_path,
            car_subtype_matrix_path=car_subtype_matrix_path,
            colour_semantic_matrix_path=config.colour_semantic_matrix_path,
            type_semantic_matrix_path=config.type_semantic_matrix_path,
            car_subtype_semantic_matrix_path=config.car_subtype_semantic_matrix_path,
            colour_classes=self.get_colour_classes(),
            type_classes=self.get_type_classes(),
            car_subtype_classes=self.get_car_subtype_classes(),
            feature_weights=config.feature_weights,
            confidence_weighting=config.confidence_weighting,
            semantic_reliability=config.semantic_reliability,
            semantic_weight=config.semantic_weight,
            adjust_for_car_type=config.adjust_for_car_type,
            p_norm_p=config.p_norm_p,
        )
        self.explicit_config = config
        self.classifier_metrics = metrics
        return metrics

    def set_cost_model(self, classifier_metrics: ClassifierMetrics) -> None:
        """Attach a prepared cost model for use by later association calculations."""
        self.classifier_metrics = classifier_metrics

    def get_cost_model(self) -> ClassifierMetrics:
        """Return the configured cost model, failing clearly if no scorer has been attached."""
        if self.classifier_metrics is None:
            raise RuntimeError(
                "Classifier metrics are not configured. Call configure_cost_from_csv(...) "
                "or set_cost_model(...) first."
            )
        return self.classifier_metrics

    def infer_embedding(self, bgr_crop: np.ndarray) -> dict[str, Any]:
        """Run the configured vehicle model on a single cropped image and return raw outputs."""
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("Invalid crop provided")
        if len(bgr_crop.shape) != 3 or bgr_crop.shape[2] != 3:
            raise ValueError(f"Crop must be (H, W, 3), got {bgr_crop.shape}")
        pil_img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        return self.model.results(pil_img)

    @staticmethod
    def crop_img(img: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
        """Crop a frame using a detection box while preserving a valid image region."""
        if img is None or box is None:
            return None
        x1, y1, x2, y2 = [int(value) for value in box]
        height, width = img.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def extract_embeddings(self, frame: np.ndarray, detections: Sequence[Any]) -> None:
        """Run the classifier on every detection crop and attach the embedding outputs in-place."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be (H, W, 3), got {frame.shape}")
        for detection in detections:
            crop = self.crop_img(frame, detection.tlbr)
            if crop is None or crop.size == 0:
                raise ValueError(f"Invalid crop for detection at {tuple(detection.tlbr)}")
            output = self.infer_embedding(crop)
            detection.explicit_embed_history.append(output)
            detection.explicit_embedding = output

    def _get_selected_cost_method(self) -> str:
        """Return the configured explicit-cost strategy name."""
        method_name = self.explicit_config.explicit_cost_method
        if not hasattr(self.get_cost_model(), method_name):
            raise AttributeError(
                f"Configured explicit_cost_method '{method_name}' is not available on "
                f"{type(self.get_cost_model()).__name__}."
            )
        return method_name

    def calculate_cost(
        self,
        track_embedding: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        detection_embedding: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> dict[str, float | None]:
        """Score a single track/detection comparison using the configured explicit feature-cost method."""
        method_name = self._get_selected_cost_method()
        method = getattr(self.get_cost_model(), method_name)
        return method(self._as_history(track_embedding), self._as_history(detection_embedding))

    def calculate_cost_matrix(
        self,
        track_embeddings: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        detection_embeddings: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        *,
        missing_cost: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        """Return directional, semantic, and per-pair attribute association costs."""
        configured_missing_cost = self.explicit_config.missing_cost if missing_cost is None else missing_cost
        track_histories = [self._as_history(value) for value in track_embeddings]
        detection_histories = [self._as_history(value) for value in detection_embeddings]
        dir_costs = np.full(
            (len(track_histories), len(detection_histories)), configured_missing_cost, dtype=float
        )
        sem_costs = np.full(
            (len(track_histories), len(detection_histories)), configured_missing_cost, dtype=float
        )
        cost_details: list[list[dict[str, Any]]] = [
            [{} for _ in detection_histories] for _ in track_histories
        ]
        metrics = self.get_cost_model()
        method_name = self._get_selected_cost_method()
        method = getattr(metrics, method_name)
        for row, track_history in enumerate(track_histories):
            for column, detection_history in enumerate(detection_histories):
                cost = method(track_history, detection_history)
                directional_cost = cost.get("directional_cost") if cost is not None else None
                semantic_cost = cost.get("semantic_cost") if cost is not None else None
                cost_details[row][column] = cost or {}
                if directional_cost is not None and semantic_cost is not None:
                    dir_costs[row, column] = float(directional_cost)
                    sem_costs[row, column] = float(semantic_cost)

        return dir_costs, sem_costs, cost_details

    @staticmethod
    def _as_history(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Normalise a single embedding or embedding history into a consistent list form."""
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            history = list(value)
            if not history:
                return []
            if all(isinstance(item, Mapping) for item in history):
                return history
        raise TypeError(
            "Embedding input must be a single embedding mapping or a sequence of embedding mappings."
        )
