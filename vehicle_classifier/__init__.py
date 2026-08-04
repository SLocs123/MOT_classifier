"""Public package API for vehicle attribute checkpoint inference."""

from .low_fps_wrapper import ExplicitExtractor
from .predictor import VehicleClassifier
from .predictor import VehicleClassifierBase
from .predictor import VehicleClassifierIntermediate
from .predictor import VehicleClassifierIntermediateWithSize
from .predictor import VehicleClassifierTypeOnly
from .predictor import VehicleClassifierWithSize
from .confidence import calculate_directional_classifier_cost
from .confidence import entropy_reliability
from .confidence import normalised_js_divergence
from .metrics import ClassifierMetrics
from .metrics import PenaltyMatrices
from .metrics import configure_classifier_metrics
from .metrics import get_classifier_metrics
from .predictor import vehicle_classifier
from .predictor import vehicle_classifier_base
from .predictor import vehicle_classifier_intermediate
from .predictor import vehicle_classifier_intermediate_with_size
from .predictor import vehicle_classifier_type_only
from .predictor import vehicle_classifier_with_size

# Public aliases for downstream tracker code.
explicitExtractor = ExplicitExtractor
explicitExctractor = ExplicitExtractor

__all__ = [
    "VehicleClassifier",
    "VehicleClassifierBase",
    "VehicleClassifierWithSize",
    "VehicleClassifierIntermediate",
    "VehicleClassifierIntermediateWithSize",
    "VehicleClassifierTypeOnly",
    "ExplicitExtractor",
    "explicitExtractor",
    "explicitExctractor",
    "PenaltyMatrices",
    "ClassifierMetrics",
    "configure_classifier_metrics",
    "get_classifier_metrics",
    "calculate_directional_classifier_cost",
    "normalised_js_divergence",
    "entropy_reliability",
    "vehicle_classifier",
    "vehicle_classifier_base",
    "vehicle_classifier_with_size",
    "vehicle_classifier_intermediate",
    "vehicle_classifier_intermediate_with_size",
    "vehicle_classifier_type_only",
]
