import cv2
from PIL import Image
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from .metrics import ClassifierMetrics
from .predictor import vehicle_classifier_intermediate




class ExplicitExtractor:
    def __init__(
        self,
        checkpoints_path: str | Path | None = None,
        dinov2_path: str | Path = "dinov2",
        classifier: Any | None = None,
        classifier_metrics: ClassifierMetrics | None = None,
    ) -> None:
        if classifier is not None:
            self.model = classifier
        else:
            if checkpoints_path is None:
                raise ValueError("checkpoints_path is required when classifier is not provided")
            self._setup(checkpoints_path, dinov2_path)
        self.classifier_metrics = classifier_metrics
    
    
    def _setup(self, checkpoints_path: str | Path, dinov2_path: str | Path):
        model = vehicle_classifier_intermediate(
            checkpoint_path=checkpoints_path,
            dinov2_repo_path=dinov2_path,
        )
        self.model = model

    def get_classifier_model(self):
        return self.model

    def get_colour_classes(self):
        return list(self.model.colour_classes)

    def get_type_classes(self):
        return list(self.model.type_classes)

    def get_car_subtype_classes(self):
        return list(self.model.car_subtype_classes)

    def configure_cost_from_csv(
        self,
        *,
        colour_matrix_path: str | Path,
        type_matrix_path: str | Path,
        car_subtype_matrix_path: str | Path,
        js_weight: float = 0.5,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool = True,
    ) -> ClassifierMetrics:
        metrics = ClassifierMetrics.from_csv_files(
            colour_matrix_path=colour_matrix_path,
            type_matrix_path=type_matrix_path,
            car_subtype_matrix_path=car_subtype_matrix_path,
            colour_classes=self.get_colour_classes(),
            type_classes=self.get_type_classes(),
            car_subtype_classes=self.get_car_subtype_classes(),
            js_weight=js_weight,
            feature_weights=feature_weights,
            confidence_weighting=confidence_weighting,
        )
        self.classifier_metrics = metrics
        return metrics

    def set_cost_model(self, classifier_metrics: ClassifierMetrics) -> None:
        self.classifier_metrics = classifier_metrics

    def get_cost_model(self) -> ClassifierMetrics:
        if self.classifier_metrics is None:
            raise RuntimeError(
                "Classifier metrics are not configured. Call configure_cost_from_csv(...) "
                "or set_cost_model(...) first."
            )
        return self.classifier_metrics

    def infer_embedding(self, bgr_crop: np.ndarray) -> dict[str, Any]:
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("Invalid crop provided")
        if len(bgr_crop.shape) != 3 or bgr_crop.shape[2] != 3:
            raise ValueError(f"Crop must be (H, W, 3), got {bgr_crop.shape}")

        pil_img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        return self.model.results(pil_img)

    def crop_img(self, img, box):
        """
        Crop one image using an xyxy box and return the raw image crop.
        """

        if img is None or box is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = img.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]
        return crop

    def extract_embeddings(self, frame, detections):
        '''
        Extract embeddings for detections in a single frame.
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            detections: list of STrack objects (new detections)
        
        Returns:
            None (modifies STrack objects in-place, appending to embed_history)
        '''

        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be (H, W, 3), got {frame.shape}")
        
        # Process detections
        for det in detections:
            x1, y1, x2, y2 = det.tlbr
            
            crop = self.crop_img(frame, (x1, y1, x2, y2))
            if crop is None or crop.size == 0:
                raise ValueError(
                    f"Invalid crop for detection at {(x1, y1, x2, y2)}"
                )
            
            out = self.infer_embedding(crop)
            """
            format:
            {
                "colour": {"index": int,
                           "label": str,
                           "confidence": float,
                           "probs": typ},
                "type": ...,
                "car_subtype": ...,
            }
            """          
            
            # explicit definition of features
            det.embed_history.append(out)
            det.embedding = out

    def calculate_cost(
        self,
        track_embedding: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        detection_embedding: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> float | None:
        metrics = self.get_cost_model()
        track_history = self._as_history(track_embedding)
        detection_history = self._as_history(detection_embedding)
        return metrics.explicit_feature_cost(track_history, detection_history)

    def calculate_cost_matrix(
        self,
        track_embeddings: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        detection_embeddings: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        *,
        missing_cost: float = 1.0,
    ) -> np.ndarray:
        metrics = self.get_cost_model()

        track_histories = [self._as_history(value) for value in track_embeddings]
        detection_histories = [self._as_history(value) for value in detection_embeddings]

        costs = np.full((len(track_histories), len(detection_histories)), missing_cost, dtype=float)
        for i, track_history in enumerate(track_histories):
            for j, detection_history in enumerate(detection_histories):
                cost = metrics.explicit_feature_cost(track_history, detection_history)
                if cost is None:
                    costs[i, j] = missing_cost
                else:
                    costs[i, j] = float(cost)
        return costs

    @staticmethod
    def _as_history(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
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
