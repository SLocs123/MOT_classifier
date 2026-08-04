from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .confidence import calculate_directional_classifier_cost
from .confidence import normalise_probabilities


FEATURE_WEIGHTS: dict[str, float] = {
    "type": 0.65,
    "car_subtype": 0.20,
    "colour": 0.15,
}

_CLASSIFIER_METRICS: ClassifierMetrics | None = None


@dataclass(frozen=True)
class PenaltyMatrices:
    """Directional penalty matrices indexed by classifier head."""

    colour: NDArray[np.float64]
    type: NDArray[np.float64]
    car_subtype: NDArray[np.float64]

    def get(self, feature_name: str) -> NDArray[np.float64]:
        try:
            return getattr(self, feature_name)
        except AttributeError as exc:
            raise KeyError(
                f"No penalty matrix configured for feature {feature_name!r}."
            ) from exc


class ClassifierMetrics:
    """Reusable explicit classifier-cost helper for tracking association."""

    def __init__(
        self,
        penalty_matrices: PenaltyMatrices,
        *,
        js_weight: float = 0.5,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool = True,
    ) -> None:
        if not 0.0 <= js_weight <= 1.0:
            raise ValueError("js_weight must be in [0, 1].")

        self.penalty_matrices = penalty_matrices
        self.js_weight = float(js_weight)
        self.confidence_weighting = bool(confidence_weighting)

        weights = dict(FEATURE_WEIGHTS if feature_weights is None else feature_weights)
        if not weights:
            raise ValueError("At least one classifier feature weight is required.")
        if any(value < 0 for value in weights.values()):
            raise ValueError("Classifier feature weights cannot be negative.")

        weight_sum = float(sum(weights.values()))
        if weight_sum <= 0:
            raise ValueError("Classifier feature weights must sum to more than zero.")

        self.feature_weights = {
            name: float(value) / weight_sum for name, value in weights.items()
        }

    @classmethod
    def from_csv_files(
        cls,
        *,
        colour_matrix_path: str | Path,
        type_matrix_path: str | Path,
        car_subtype_matrix_path: str | Path,
        colour_classes: Sequence[str],
        type_classes: Sequence[str],
        car_subtype_classes: Sequence[str],
        js_weight: float = 0.5,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool = True,
    ) -> "ClassifierMetrics":
        matrices = PenaltyMatrices(
            colour=_read_penalty_matrix(
                colour_matrix_path,
                expected_classes=colour_classes,
            ),
            type=_read_penalty_matrix(
                type_matrix_path,
                expected_classes=type_classes,
            ),
            car_subtype=_read_penalty_matrix(
                car_subtype_matrix_path,
                expected_classes=car_subtype_classes,
            ),
        )
        return cls(
            matrices,
            js_weight=js_weight,
            feature_weights=feature_weights,
            confidence_weighting=confidence_weighting,
        )

    def explicit_feature_cost(
        self,
        track_embed_history: Sequence[Mapping[str, Any]],
        detection_embed_history: Sequence[Mapping[str, Any]],
    ) -> float | None:
        if not track_embed_history or not detection_embed_history:
            return None

        track_output = track_embed_history[-1]
        detection_output = detection_embed_history[-1]

        _validate_explicit_output(track_output, source_name="track")
        _validate_explicit_output(detection_output, source_name="detection")

        weighted_cost = 0.0
        used_weight = 0.0

        for feature_name, feature_weight in self.feature_weights.items():
            if feature_name not in track_output or feature_name not in detection_output:
                continue

            track_feature = track_output[feature_name]
            detection_feature = detection_output[feature_name]

            track_probs = normalise_probabilities(
                track_feature["probs"],
                name=f"track {feature_name} probabilities",
            )
            detection_probs = normalise_probabilities(
                detection_feature["probs"],
                name=f"detection {feature_name} probabilities",
            )

            penalty_matrix = self.penalty_matrices.get(feature_name)
            feature_cost = calculate_directional_classifier_cost(
                track_probabilities=track_probs,
                detection_probabilities=detection_probs,
                directional_penalty_matrix=penalty_matrix,
                js_weight=self.js_weight,
                apply_confidence_weighting=self.confidence_weighting,
            )

            weighted_cost += feature_weight * feature_cost
            used_weight += feature_weight

        if used_weight == 0.0:
            return None

        return float(np.clip(weighted_cost / used_weight, 0.0, 1.0))


def configure_classifier_metrics(
    *,
    colour_matrix_path: str | Path,
    type_matrix_path: str | Path,
    car_subtype_matrix_path: str | Path,
    colour_classes: Sequence[str],
    type_classes: Sequence[str],
    car_subtype_classes: Sequence[str],
    js_weight: float = 0.5,
    feature_weights: Mapping[str, float] | None = None,
    confidence_weighting: bool = True,
) -> ClassifierMetrics:
    global _CLASSIFIER_METRICS

    _CLASSIFIER_METRICS = ClassifierMetrics.from_csv_files(
        colour_matrix_path=colour_matrix_path,
        type_matrix_path=type_matrix_path,
        car_subtype_matrix_path=car_subtype_matrix_path,
        colour_classes=colour_classes,
        type_classes=type_classes,
        car_subtype_classes=car_subtype_classes,
        js_weight=js_weight,
        feature_weights=feature_weights,
        confidence_weighting=confidence_weighting,
    )
    return _CLASSIFIER_METRICS


def get_classifier_metrics() -> ClassifierMetrics:
    if _CLASSIFIER_METRICS is None:
        raise RuntimeError(
            "Classifier metrics have not been configured. Call "
            "configure_classifier_metrics(...) once during tracker setup."
        )
    return _CLASSIFIER_METRICS


def _read_penalty_matrix(
    csv_path: str | Path,
    *,
    expected_classes: Sequence[str],
) -> NDArray[np.float64]:
    expected = [str(label).strip() for label in expected_classes]
    if not expected:
        raise ValueError("expected_classes cannot be empty.")

    frame = pd.read_csv(csv_path, index_col=0)
    frame.index = frame.index.map(lambda value: str(value).strip())
    frame.columns = frame.columns.map(lambda value: str(value).strip())

    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"Duplicate labels found in penalty matrix {csv_path!s}.")

    missing_rows = sorted(set(expected) - set(frame.index))
    missing_columns = sorted(set(expected) - set(frame.columns))
    extra_rows = sorted(set(frame.index) - set(expected))
    extra_columns = sorted(set(frame.columns) - set(expected))

    if missing_rows or missing_columns or extra_rows or extra_columns:
        raise ValueError(
            f"Penalty-matrix labels do not match classifier classes for {csv_path!s}. "
            f"Missing rows={missing_rows}, missing columns={missing_columns}, "
            f"extra rows={extra_rows}, extra columns={extra_columns}."
        )

    frame = frame.loc[expected, expected]

    try:
        matrix = frame.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Penalty matrix {csv_path!s} contains non-numeric values."
        ) from exc

    expected_shape = (len(expected), len(expected))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Penalty matrix {csv_path!s} has shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Penalty matrix {csv_path!s} contains NaN or infinity.")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(
            f"Penalty matrix {csv_path!s} contains values outside [0, 1]."
        )
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        raise ValueError(
            f"Penalty matrix {csv_path!s} must have a zero diagonal."
        )

    return matrix


def _validate_explicit_output(
    output: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    if not isinstance(output, Mapping):
        raise TypeError(
            f"Latest {source_name} embedding must be a mapping, "
            f"received {type(output).__name__}."
        )

    for feature_name in ("colour", "type", "car_subtype"):
        if feature_name not in output:
            continue

        feature_data = output[feature_name]
        if not isinstance(feature_data, Mapping):
            raise TypeError(
                f"{source_name} feature {feature_name!r} must be a mapping."
            )
        if "probs" not in feature_data:
            raise KeyError(
                f"{source_name} feature {feature_name!r} is missing 'probs'."
            )
