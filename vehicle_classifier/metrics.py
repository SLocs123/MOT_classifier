from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .confidence import calculate_directional_classifier_cost
from .confidence import calculate_semantic_classifier_cost
from .confidence import normalise_probabilities


FEATURE_WEIGHTS: dict[str, float] = {
    "type": 0.65,
    "car_subtype": 0.20,
    "colour": 0.15,
}

_CLASSIFIER_METRICS: ClassifierMetrics | None = None


def _compute_min_cosine_similarity(matrix: NDArray[np.float64]) -> float:
    """Return the lowest cosine similarity between any two distinct rows of a semantic penalty matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalised_rows = matrix / safe_norms
    cosine = normalised_rows @ normalised_rows.T
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    off_diagonal = cosine[~np.eye(n, dtype=bool)]
    return float(np.clip(off_diagonal.min(), 0.0, 1.0))


@dataclass(frozen=True)
class PenaltyMatrices:
    """Directional penalty matrices indexed by classifier head."""

    colour: NDArray[np.float64]
    type: NDArray[np.float64]
    car_subtype: NDArray[np.float64]
    colour_semantic: NDArray[np.float64] | None = None
    type_semantic: NDArray[np.float64] | None = None
    car_subtype_semantic: NDArray[np.float64] | None = None
    semantic_min_cos: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Calibrate each semantic matrix's achievable cosine-similarity floor for cost rescaling."""
        min_cos: dict[str, float] = {}
        for feature_name in ("colour", "type", "car_subtype"):
            semantic_matrix = getattr(self, f"{feature_name}_semantic")
            if semantic_matrix is not None:
                min_cos[feature_name] = _compute_min_cosine_similarity(semantic_matrix)
        object.__setattr__(self, "semantic_min_cos", min_cos)

    def get(self, feature_name: str) -> NDArray[np.float64]:
        """Return the matrix associated with a named feature in the classifier cost model."""
        try:
            return getattr(self, feature_name)
        except AttributeError as exc:
            raise KeyError(
                f"No penalty matrix configured for feature {feature_name!r}."
            ) from exc

    def get_semantic_min_cos(self, feature_name: str) -> float | None:
        """Return the calibrated cosine-similarity floor for a feature's semantic matrix, if computed."""
        return self.semantic_min_cos.get(feature_name)


def cosine_similarity(vector_a: ArrayLike, vector_b: ArrayLike, *, eps: float = 1e-8) -> float | None:
    """Return cosine similarity for two feature vectors, or None when either vector is effectively empty."""
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < eps or norm_b < eps:
        return None
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_feature_distance(vector_a: ArrayLike, vector_b: ArrayLike, *, eps: float = 1e-8) -> float | None:
    """Convert cosine similarity into a [0, 1] feature distance for appearance-based matching."""
    similarity = cosine_similarity(vector_a, vector_b, eps=eps)
    if similarity is None:
        return None
    return float(1.0 - (similarity + 1.0) / 2.0)


class ClassifierMetrics:
    """Reusable explicit classifier-cost helper for tracking association."""

    def __init__(
        self,
        penalty_matrices: PenaltyMatrices,
        *,
        js_weight: float = 0.5,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool = True,
        semantic_reliability: bool = True,
        semantic_weight: float = 0.6,
        adjust_for_car_type: bool = True,
        p_norm_p: float = 2.0,
    ) -> None:
        """Store the cost model, feature weights, and semantic weighting used during association scoring."""
        if not 0.0 <= js_weight <= 1.0:
            raise ValueError("js_weight must be in [0, 1].")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1].")
        if p_norm_p < 1.0:
            raise ValueError("p_norm_p must be >= 1.")

        self.penalty_matrices = penalty_matrices
        self.js_weight = float(js_weight)
        self.confidence_weighting = bool(confidence_weighting)
        self.semantic_reliability = bool(semantic_reliability)
        self.semantic_weight = float(semantic_weight)
        self.adjust_for_car_type = bool(adjust_for_car_type)
        self.p_norm_p = float(p_norm_p)

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
        colour_semantic_matrix_path: str | Path | None = None,
        type_semantic_matrix_path: str | Path | None = None,
        car_subtype_semantic_matrix_path: str | Path | None = None,
        colour_classes: Sequence[str],
        type_classes: Sequence[str],
        car_subtype_classes: Sequence[str],
        js_weight: float = 0.5,
        feature_weights: Mapping[str, float] | None = None,
        confidence_weighting: bool = True,
        semantic_reliability: bool = True,
        semantic_weight: float = 0.6,
        adjust_for_car_type: bool = True,
        p_norm_p: float = 2.0,
    ) -> "ClassifierMetrics":
        """Construct a cost model from CSV penalty matrices and classifier label metadata."""
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
            colour_semantic=_read_penalty_matrix(
                colour_semantic_matrix_path,
                expected_classes=colour_classes,
            ),
            type_semantic=_read_penalty_matrix(
                type_semantic_matrix_path,
                expected_classes=type_classes,
            ),
            car_subtype_semantic=_read_penalty_matrix(
                car_subtype_semantic_matrix_path,
                expected_classes=car_subtype_classes,
            ),
        )
        return cls(
            matrices,
            js_weight=js_weight,
            feature_weights=feature_weights,
            confidence_weighting=confidence_weighting,
            semantic_reliability=semantic_reliability,
            semantic_weight=semantic_weight,
            adjust_for_car_type=adjust_for_car_type,
            p_norm_p=p_norm_p,
        )

    def adjust_feature_weights_for_car_type(
        self,
        track_output: Mapping[str, Any],
        detection_output: Mapping[str, Any],
    ) -> dict[str, float]:
        """Reduce type weight when both predictions are confident car labels and keep weights renormalised."""
        
        if not self.adjust_for_car_type:
            return dict(self.feature_weights)

        adjusted_weights = dict(self.feature_weights)
        track_car = (
            track_output.get("type", {}).get("label", "").lower() == "vehicle.car"
            and float(track_output["type"].get("confidence", 0.0)) > 0.65
        )
        detection_car = (
            detection_output.get("type", {}).get("label", "").lower() == "vehicle.car"
            and float(detection_output["type"].get("confidence", 0.0)) > 0.65
        )

        if not (track_car and detection_car):
            original_car_subtype_weight = adjusted_weights.get("car_subtype", 0.0)
            adjusted_weights["car_subtype"] = 0.0 # set car_subtype to 0 if not confident car.
            adjusted_weights["type"] = max(0.0, adjusted_weights.get("type", 0.0) + original_car_subtype_weight)
        else:
            adjusted_weights["car_subtype"] += adjusted_weights.get("type", 0.0) * 0.5 # set car_subtype to increase by half of type weight if both are confident car.
            adjusted_weights["type"] *= 0.5 # downweight type by half
            

        weight_sum = sum(adjusted_weights.values())
        if weight_sum > 0.0:
            adjusted_weights = {name: value / weight_sum for name, value in adjusted_weights.items()}
        return adjusted_weights

    def explicit_feature_cost(
        self,
        track_embed_history: Sequence[Mapping[str, Any]],
        detection_embed_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Compare the latest track and detection embeddings across each classifier feature."""
        
        # method compose to type and colour values, then use p-norm to combine. or does p-norm inherenty handle the type issue, when a larger car error, will naturally have more of an effect than the tiny/0 type error?
        # Maybe best to test all approaches so, 1. Current, weighted average with downweighted car and subtype when car, 2. p-norm where all values are combined, 3. p-norm where type and subtype are combined first
        # This will likely need a new explicit_feature_cost for each method, check if good way to combine into 1
        if not track_embed_history or not detection_embed_history:
            return {"directional_cost": None, "semantic_cost": None}

        track_output = track_embed_history[-1]
        detection_output = detection_embed_history[-1]

        _validate_explicit_output(track_output, source_name="track")
        _validate_explicit_output(detection_output, source_name="detection")

        weighted_directional_cost = 0.0
        weighted_semantic_cost = 0.0
        used_weight = 0.0
        attribute_costs: dict[str, dict[str, float]] = {}

        effective_weights = self.adjust_feature_weights_for_car_type(track_output, detection_output)

        for feature_name, feature_weight in effective_weights.items():
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
            directional_cost = calculate_directional_classifier_cost(
                track_probabilities=track_probs,
                detection_probabilities=detection_probs,
                directional_penalty_matrix=penalty_matrix,
                semantic_weight=self.semantic_weight,
                apply_confidence_weighting=self.confidence_weighting,
                semantic_reliability=self.semantic_reliability,
            )

            semantic_matrix_name = f"{feature_name}_semantic"
            semantic_matrix = getattr(self.penalty_matrices, semantic_matrix_name, None)
            if semantic_matrix is not None:
                semantic_cost = calculate_semantic_classifier_cost(
                    track_probabilities=track_probs,
                    detection_probabilities=detection_probs,
                    semantic_penalty_matrix=semantic_matrix,
                    semantic_weight=self.semantic_weight,
                    apply_confidence_weighting=self.confidence_weighting,
                    semantic_reliability=self.semantic_reliability,
                )
            else:
                semantic_cost = directional_cost

            directional_contribution = feature_weight * directional_cost
            semantic_contribution = feature_weight * semantic_cost
            attribute_costs[feature_name] = {
                "weight": float(feature_weight),
                "directional_cost": float(directional_cost),
                "semantic_cost": float(semantic_cost),
                "directional_contribution": float(directional_contribution),
                "semantic_contribution": float(semantic_contribution),
            }
            weighted_directional_cost += directional_contribution
            weighted_semantic_cost += semantic_contribution
            used_weight += feature_weight

        if used_weight == 0.0:
            return {"directional_cost": None, "semantic_cost": None, "attribute_costs": {}}

        return {
            "directional_cost": float(np.clip(weighted_directional_cost / used_weight, 0.0, 1.0)),
            "semantic_cost": float(np.clip(weighted_semantic_cost / used_weight, 0.0, 1.0)),
            "attribute_costs": attribute_costs,
        }

    def explicit_feature_cost_p_norm(
        self,
        track_embed_history: Sequence[Mapping[str, Any]],
        detection_embed_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Compare the latest track and detection embeddings across each classifier feature. using p-norm aggregation of costs across features."""
        
        # My concern with this method is that a error in car_subtype, which is a difficult classification, might spike the p-norm, Must ensure that semantic_matrix correctly represents the car_subtype difficulty, i.e. doesnt punish too much for car error.
        if not track_embed_history or not detection_embed_history:
            return {"directional_cost": None, "semantic_cost": None}

        track_output = track_embed_history[-1]
        detection_output = detection_embed_history[-1]

        _validate_explicit_output(track_output, source_name="track")
        _validate_explicit_output(detection_output, source_name="detection")

        head_costs = {}
        
        # check if both are confident car, if so downweight type and upweight subtype, then normalise weights
        effective_weights = self.adjust_feature_weights_for_car_type(track_output, detection_output)
        if effective_weights['car_subtype'] == 0.0: # not a car
            skip_subtype = True
        else:
            skip_subtype = False

        for feature_name, feature_weight in self.feature_weights.items():
            if feature_name not in track_output or feature_name not in detection_output:
                continue
            
            if skip_subtype and feature_name == "car_subtype":
                continue # skip car_subtype if not a car, can still include type cost for car vs car because p-norm will handle the small loss non-compensatorily
            
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
            directional_cost = calculate_directional_classifier_cost(
                track_probabilities=track_probs,
                detection_probabilities=detection_probs,
                directional_penalty_matrix=penalty_matrix,
                semantic_weight=self.semantic_weight,
                apply_confidence_weighting=self.confidence_weighting,
                semantic_reliability=self.semantic_reliability,
            )

            semantic_matrix_name = f"{feature_name}_semantic"
            semantic_matrix = getattr(self.penalty_matrices, semantic_matrix_name, None)
            if semantic_matrix is not None:
                semantic_cost = calculate_semantic_classifier_cost(
                    track_probabilities=track_probs,
                    detection_probabilities=detection_probs,
                    semantic_penalty_matrix=semantic_matrix,
                    semantic_weight=self.semantic_weight,
                    apply_confidence_weighting=self.confidence_weighting,
                    semantic_reliability=self.semantic_reliability,
                )
            else:
                semantic_cost = directional_cost
            

            head_costs[feature_name] = {
                "directional_cost": directional_cost,
                "semantic_cost": semantic_cost,
            }
            
        # Weighted power-mean (generalised mean) aggregation: (sum(w_i * c_i^p) / sum(w_i)) ** (1/p).
        # For p=1 this is the plain weighted average; for p>1 it is >= the weighted average
        # (power-mean inequality) and tends to max(c_i) as p grows, so a single bad cost is
        # non-compensatorily amplified rather than diluted by the other, good, costs.
        p = self.p_norm_p
        used_weight = sum(self.feature_weights[feature_name] for feature_name in head_costs)
        weighted_directional_cost = (sum(
            self.feature_weights[feature_name] * head_costs[feature_name]["directional_cost"] ** p
            for feature_name in head_costs
        ) / used_weight) ** (1 / p)
        weighted_semantic_cost = (sum(
            self.feature_weights[feature_name] * head_costs[feature_name]["semantic_cost"] ** p
            for feature_name in head_costs
        ) / used_weight) ** (1 / p)

        attribute_costs = {
            feature_name: {
                "weight": float(self.feature_weights[feature_name]),
                "directional_cost": float(costs["directional_cost"]),
                "semantic_cost": float(costs["semantic_cost"]),
                # contribution = weight * raw^p (the term summed before taking the 1/p root).
                "directional_contribution": float(self.feature_weights[feature_name] * costs["directional_cost"] ** p),
                "semantic_contribution": float(self.feature_weights[feature_name] * costs["semantic_cost"] ** p),
            }
            for feature_name, costs in head_costs.items()
        }

        return {
            "directional_cost": float(np.clip(weighted_directional_cost, 0.0, 1.0)),
            "semantic_cost": float(np.clip(weighted_semantic_cost, 0.0, 1.0)),
            "attribute_costs": attribute_costs,
        }

    def explicit_feature_cost_p_norm_combined(
    self,
    track_embed_history: Sequence[Mapping[str, Any]],
    detection_embed_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Compare the latest track and detection embeddings across each classifier feature. Using p-norm aggregation of costs across features, with type and subtype combined first."""
        
        if not track_embed_history or not detection_embed_history:
            return {"directional_cost": None, "semantic_cost": None}

        track_output = track_embed_history[-1]
        detection_output = detection_embed_history[-1]

        _validate_explicit_output(track_output, source_name="track")
        _validate_explicit_output(detection_output, source_name="detection")

        head_costs = {}
        
        # check if both are confident car, if so downweight type and upweight subtype, then normalise weights
        effective_weights = self.adjust_feature_weights_for_car_type(track_output, detection_output)


        for feature_name, feature_weight in effective_weights.items():
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
            directional_cost = calculate_directional_classifier_cost(
                track_probabilities=track_probs,
                detection_probabilities=detection_probs,
                directional_penalty_matrix=penalty_matrix,
                semantic_weight=self.semantic_weight,
                apply_confidence_weighting=self.confidence_weighting,
                semantic_reliability=self.semantic_reliability,
            )

            semantic_matrix_name = f"{feature_name}_semantic"
            semantic_matrix = getattr(self.penalty_matrices, semantic_matrix_name, None)
            if semantic_matrix is not None:
                semantic_cost = calculate_semantic_classifier_cost(
                    track_probabilities=track_probs,
                    detection_probabilities=detection_probs,
                    semantic_penalty_matrix=semantic_matrix,
                    semantic_weight=self.semantic_weight,
                    apply_confidence_weighting=self.confidence_weighting,
                    semantic_reliability=self.semantic_reliability,
                )
            else:
                semantic_cost = directional_cost
            

            head_costs[feature_name] = {
                "directional_cost": directional_cost,
                "semantic_cost": semantic_cost,
            }
            
            
        # combine type and subtype before p-norm aggregation
        combined_weight = effective_weights['type'] + effective_weights['car_subtype']
        if combined_weight > 0.0:
            combined_directional_cost = (
                head_costs['type']["directional_cost"] * effective_weights['type']
                + head_costs['car_subtype']["directional_cost"] * effective_weights['car_subtype']
            ) / combined_weight
            combined_semantic_cost = (
                head_costs['type']["semantic_cost"] * effective_weights['type']
                + head_costs['car_subtype']["semantic_cost"] * effective_weights['car_subtype']
            ) / combined_weight
        else:
            combined_directional_cost = head_costs['type']["directional_cost"]
            combined_semantic_cost = head_costs['type']["semantic_cost"]
        head_costs["type_combined"] = {
            "directional_cost": combined_directional_cost,
            "semantic_cost": combined_semantic_cost,
        }
        del head_costs['type']
        del head_costs['car_subtype']
        effective_weights['type_combined'] = combined_weight
        
        # Weighted power-mean (generalised mean) aggregation: (sum(w_i * c_i^p) / sum(w_i)) ** (1/p).
        # For p=1 this is the plain weighted average; for p>1 it is >= the weighted average
        # (power-mean inequality) and tends to max(c_i) as p grows, so a single bad cost is
        # non-compensatorily amplified rather than diluted by the other, good, costs.
        p = self.p_norm_p
        used_weight = sum(effective_weights[feature_name] for feature_name in head_costs)
        weighted_directional_cost = (sum(
            effective_weights[feature_name] * head_costs[feature_name]["directional_cost"] ** p
            for feature_name in head_costs
        ) / used_weight) ** (1 / p)
        weighted_semantic_cost = (sum(
            effective_weights[feature_name] * head_costs[feature_name]["semantic_cost"] ** p
            for feature_name in head_costs
        ) / used_weight) ** (1 / p)

        attribute_costs = {
            feature_name: {
                "weight": float(effective_weights[feature_name]),
                "directional_cost": float(costs["directional_cost"]),
                "semantic_cost": float(costs["semantic_cost"]),
                # contribution = weight * raw^p (the term summed before taking the 1/p root).
                "directional_contribution": float(effective_weights[feature_name] * costs["directional_cost"] ** p),
                "semantic_contribution": float(effective_weights[feature_name] * costs["semantic_cost"] ** p),
            }
            for feature_name, costs in head_costs.items()
        }

        return {
            "directional_cost": float(np.clip(weighted_directional_cost, 0.0, 1.0)),
            "semantic_cost": float(np.clip(weighted_semantic_cost, 0.0, 1.0)),
            "attribute_costs": attribute_costs,
        }
    
    def cosine_feature_cost(
        self,
        track_embed_history: Sequence[Mapping[str, Any]],
        detection_embed_history: Sequence[Mapping[str, Any]],
    ) -> float | None:
        """Compare average backbone embeddings by cosine distance as a fallback appearance cost."""
        track_vector = _mean_feature_vector(track_embed_history)
        detection_vector = _mean_feature_vector(detection_embed_history)
        if track_vector is None or detection_vector is None:
            return None
        return cosine_feature_distance(track_vector, detection_vector)


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
    semantic_reliability: bool = True,
    semantic_weight: float = 0.6,
    adjust_for_car_type: bool = True,
    p_norm_p: float = 2.0,
) -> ClassifierMetrics:
    """Create and store the module-level cost model used by the tracker during association."""
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
        semantic_reliability=semantic_reliability,
        semantic_weight=semantic_weight,
        adjust_for_car_type=adjust_for_car_type,
        p_norm_p=p_norm_p,
    )
    return _CLASSIFIER_METRICS


def get_classifier_metrics() -> ClassifierMetrics:
    """Return the configured global cost model or raise a clear error if it has not been initialized."""
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
    """Load and validate a directional penalty matrix against the classifier label ordering."""
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


def _mean_feature_vector(
    embed_history: Sequence[Mapping[str, Any]],
) -> NDArray[np.float64] | None:
    """Average the stored feature vectors over time for cosine-distance comparisons."""
    vectors = [
        np.asarray(entry["feature_vector"], dtype=np.float64)
        for entry in embed_history
        if isinstance(entry, Mapping) and "feature_vector" in entry
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _validate_explicit_output(
    output: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    """Ensure a classifier output contains the expected probability-bearing feature dictionaries."""
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
