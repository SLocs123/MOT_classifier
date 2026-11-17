from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _to_numpy_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert model outputs into a validated 1-D NumPy array for probability comparisons."""
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
    """Scale a probability vector to sum to one without changing the relative class weights."""
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
    """Measure the normalised Jensen-Shannon disagreement between two class-probability vectors."""
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
    """Return a confidence-like reliability score based on prediction entropy."""
    probs = normalise_probabilities(probabilities, name="probabilities")

    class_count = probs.size
    if class_count <= 1:
        return 1.0

    positive = probs > 0.0
    entropy = -np.sum(probs[positive] * np.log(np.maximum(probs[positive], epsilon)))
    reliability = 1.0 - entropy / np.log(class_count)
    return float(np.clip(reliability, 0.0, 1.0))

def semantic_reliability(p, M):
    """Estimate the reliability of a single probability vector from the matrix-induced semantic cost."""
    u = float(p @ M @ p)
    r = 1.0 - u
    return r

def pair_reliability(p, q, M):
    """Combine the semantic reliability of a pair into a single score for comparison weighting."""
    rp = semantic_reliability(p, M)
    rq = semantic_reliability(q, M)
    return float(np.sqrt(rp * rq))

def calculate_directional_classifier_cost(
    *,
    track_probabilities: ArrayLike,
    detection_probabilities: ArrayLike,
    directional_penalty_matrix: ArrayLike,
    semantic_weight: float = 0.6,
    apply_confidence_weighting: bool = True,
    semantic_reliability: bool = True,
) -> float:
    """Blend matrix disagreement and probability divergence into a single association cost for a track/detection pair."""
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be in [0, 1].")

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

    if apply_confidence_weighting:
        if semantic_reliability:
            r_pair = pair_reliability(track_probs, detection_probs, penalty_matrix)
            reliability = r_pair
        else:
            track_reliability = entropy_reliability(track_probs)
            detection_reliability = entropy_reliability(detection_probs)
            comparison_reliability = np.sqrt(track_reliability * detection_reliability)
            reliability = comparison_reliability
            
        # classifier_cost = semantic_weight * matrix_cost + (1.0 - semantic_weight) * (reliability * js_cost)
        classifier_cost = matrix_cost # remove js for testing
    else:
        # classifier_cost = semantic_weight * matrix_cost + (1.0 - semantic_weight) * js_cost
        classifier_cost = matrix_cost # remove js for testing
    return float(np.clip(classifier_cost, 0.0, 1.0))


def calculate_semantic_classifier_cost(
    *,
    track_probabilities: ArrayLike,
    detection_probabilities: ArrayLike,
    semantic_penalty_matrix: ArrayLike,
    semantic_weight: float = 0.6,
    apply_confidence_weighting: bool = True,
    semantic_reliability: bool = True,
    use_semantic_vector: bool = False,
    min_cosine_similarity: float | None = None,
) -> float:
    """Similar to calculate_directional_classifier_cost but uses semantic/symetrical penalty matrices instead of directional ones.

    Uses the direct bilinear form E[M[i,j]] over the two probability distributions (same mechanism as
    calculate_directional_classifier_cost). This was previously computed via cosine similarity between
    each distribution's projected "semantic vector" (use_semantic_vector=True) - that method has been
    removed (see the commented-out block below) because it measured whether the two distributions'
    row-dissimilarity PROFILES pointed in a similar direction, not the actual expected cost between the
    predicted classes, and its achievable range was capped by the matrix's row geometry/dimensionality
    rather than by its authored cost magnitudes.
    """
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be in [0, 1].")

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

    penalty_matrix = np.asarray(semantic_penalty_matrix, dtype=np.float64)
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

    if not use_semantic_vector:
        matrix_cost = float(track_probs @ penalty_matrix @ detection_probs)
        # js_cost = normalised_js_divergence(track_probs, detection_probs)

        if apply_confidence_weighting:
            if semantic_reliability:
                r_pair = pair_reliability(track_probs, detection_probs, penalty_matrix)
                reliability = r_pair
            else:
                track_reliability = entropy_reliability(track_probs)
                detection_reliability = entropy_reliability(detection_probs)
                comparison_reliability = np.sqrt(track_reliability * detection_reliability)
                reliability = comparison_reliability
                
            # classifier_cost = semantic_weight * matrix_cost + (1.0 - semantic_weight) * (reliability * js_cost)
            classifier_cost = matrix_cost # remove js for testing
        else:
            # classifier_cost = semantic_weight * matrix_cost + (1.0 - semantic_weight) * js_cost
            classifier_cost = matrix_cost # remove js for testing
    else:
        # REMOVED: cosine-similarity-of-semantic-vector method. This projected each probability vector
        # through the penalty matrix into a "semantic vector" (a row-profile of dissimilarities to every
        # other class), then compared the two profiles via cosine similarity. That measures whether the
        # two distributions' dissimilarity PROFILES point in a similar direction, not the actual expected
        # cost between the predicted classes, and its achievable range was capped by the matrix's row
        # geometry/dimensionality rather than by its authored cost magnitudes. Kept here for reference:
        #
        # track_semantic_vector = track_probs @ penalty_matrix
        # detection_semantic_vector = detection_probs @ penalty_matrix
        #
        # # cosine similarity of vectors in semantic space
        # dot_product = float(track_semantic_vector @ detection_semantic_vector)
        # track_norm = float(np.linalg.norm(track_semantic_vector))
        # detection_norm = float(np.linalg.norm(detection_semantic_vector))
        # if track_norm <= 0.0 or detection_norm <= 0.0:
        #     cosine_similarity = 0.0
        # else:
        #     cosine_similarity = dot_product / (track_norm * detection_norm)
        #
        # if min_cosine_similarity is not None and min_cosine_similarity < 1.0:
        #     # Rescale so the matrix's true achievable range ([min_cosine_similarity, 1.0]) maps to [0, 1],
        #     # instead of the raw cost being capped by however similar the matrix's rows are structurally.
        #     matrix_cost = (1.0 - cosine_similarity) / (1.0 - min_cosine_similarity)
        # else:
        #     matrix_cost = 1.0 - cosine_similarity  # convert similarity to cost
        #
        # classifier_cost = matrix_cost
        raise NotImplementedError(
            "use_semantic_vector=True has been removed; it did not faithfully represent the matrix's "
            "intended costs (see the commented-out block above). Use the default bilinear form instead."
        )
    
    return float(np.clip(classifier_cost, 0.0, 1.0))
