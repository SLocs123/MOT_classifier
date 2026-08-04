from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _to_numpy_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert lists/arrays/tensors to a clean 1-D float64 array."""
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()

    array = np.asarray(values, dtype=np.float64)
    array = np.squeeze(array)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} contains negative values.")

    return array


def normalise_probabilities(probabilities: ArrayLike, *, name: str) -> NDArray[np.float64]:
    probs = _to_numpy_1d(probabilities, name=name)
    total = float(probs.sum())

    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum.")

    return probs / total


def normalised_js_divergence(
    track_probabilities: ArrayLike,
    detection_probabilities: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    track_probs = normalise_probabilities(track_probabilities, name="track_probabilities")
    detection_probs = normalise_probabilities(
        detection_probabilities,
        name="detection_probabilities",
    )

    if track_probs.size != detection_probs.size:
        raise ValueError(
            "Track and detection probability vectors have different lengths: "
            f"{track_probs.size} and {detection_probs.size}."
        )

    mixture = 0.5 * (track_probs + detection_probs)

    track_mask = track_probs > 0.0
    detection_mask = detection_probs > 0.0

    track_kl = np.sum(
        track_probs[track_mask]
        * np.log(track_probs[track_mask] / np.maximum(mixture[track_mask], epsilon))
    )
    detection_kl = np.sum(
        detection_probs[detection_mask]
        * np.log(
            detection_probs[detection_mask]
            / np.maximum(mixture[detection_mask], epsilon)
        )
    )

    js_divergence = 0.5 * (track_kl + detection_kl)
    return float(np.clip(js_divergence / np.log(2.0), 0.0, 1.0))


def entropy_reliability(
    probabilities: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    probs = normalise_probabilities(probabilities, name="probabilities")

    class_count = probs.size
    if class_count <= 1:
        return 1.0

    positive = probs > 0.0
    entropy = -np.sum(probs[positive] * np.log(np.maximum(probs[positive], epsilon)))
    reliability = 1.0 - entropy / np.log(class_count)
    return float(np.clip(reliability, 0.0, 1.0))


def calculate_directional_classifier_cost(
    *,
    track_probabilities: ArrayLike,
    detection_probabilities: ArrayLike,
    directional_penalty_matrix: ArrayLike,
    js_weight: float,
    apply_confidence_weighting: bool,
) -> float:
    """
    Blend Jensen-Shannon divergence and a directional matrix cost.

    Matrix direction is intentional:
    track probabilities select rows (previous state), detection probabilities
    select columns (current state).
    """
    if not 0.0 <= js_weight <= 1.0:
        raise ValueError("js_weight must be in [0, 1].")

    track_probs = normalise_probabilities(
        track_probabilities,
        name="track_probabilities",
    )
    detection_probs = normalise_probabilities(
        detection_probabilities,
        name="detection_probabilities",
    )

    class_count = track_probs.size
    if detection_probs.size != class_count:
        raise ValueError(
            "Track and detection probability vectors have different lengths: "
            f"{class_count} and {detection_probs.size}."
        )

    penalty_matrix = np.asarray(directional_penalty_matrix, dtype=np.float64)
    expected_shape = (class_count, class_count)
    if penalty_matrix.shape != expected_shape:
        raise ValueError(
            "Probability-vector length does not match the penalty matrix: "
            f"vector length={class_count}, matrix shape={penalty_matrix.shape}."
        )
    if not np.all(np.isfinite(penalty_matrix)):
        raise ValueError("Penalty matrix contains NaN or infinity.")
    if np.any((penalty_matrix < 0.0) | (penalty_matrix > 1.0)):
        raise ValueError("Penalty matrix contains values outside [0, 1].")

    matrix_cost = float(track_probs @ penalty_matrix @ detection_probs)
    js_cost = normalised_js_divergence(track_probs, detection_probs)

    classifier_cost = js_weight * js_cost + (1.0 - js_weight) * matrix_cost

    if not apply_confidence_weighting:
        return float(np.clip(classifier_cost, 0.0, 1.0))

    track_reliability = entropy_reliability(track_probs)
    detection_reliability = entropy_reliability(detection_probs)
    comparison_reliability = np.sqrt(track_reliability * detection_reliability)

    return float(np.clip(comparison_reliability * classifier_cost, 0.0, 1.0))
