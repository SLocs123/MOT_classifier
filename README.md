# MOT_classifier

Inference and explicit-feature costing for the vehicle attribute checkpoints
produced by this repository's DINOv2 trainers.

## Install

From this repository root:

```bash
pip install -e MOT_classifier
```

## Package layout

The implementation is split by responsibility:

- `model_loading.py` loads checkpoints, metadata, DINOv2 backbones, and model variants.
- `modeling.py` defines the neural-network architectures.
- `predictor.py` performs image preprocessing and classifier inference. Its historical factory functions remain public.
- `config.py` contains the three configuration objects: `ClassificationConfig` for preprocessing, `ExplicitFeatureConfig` for costing, and `ModelConfig` for model loading.
- `feature_extraction.py` crops detections, runs explicit attribute inference, and delegates cost calculations.
- `config.py` contains penalty-matrix paths, feature weights, JS weighting, and missing-cost settings through `ExplicitFeatureConfig`.
- `metrics.py` loads penalty matrices and aggregates per-feature costs.
- `confidence.py` contains probability normalization, reliability, and directional confidence calculations.
- `ExplicitExtractor` remains a package-level alias for `ExplicitFeatureExtractor` for downstream compatibility.

## Classifier usage

```python
from vehicle_classifier import vehicle_classifier

model = vehicle_classifier("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
out = model.results("/path/to/image.jpg")
print(out)
```

`vehicle_classifier(...)` auto-detects which checkpoint architecture it is loading.

Explicit entry points for each experimental trainer are also available:

`vehicle_classifier(...)` auto-detects the checkpoint architecture. Explicit
factories are also available: `vehicle_classifier_base`,
`vehicle_classifier_with_size`, `vehicle_classifier_intermediate`,
`vehicle_classifier_intermediate_with_size`, and `vehicle_classifier_type_only`.

The original calling process is still supported:

```python
from vehicle_classifier import vehicle_classifier

model = vehicle_classifier(
  "/path/to/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
```

## Configuration

There are three configuration objects in `vehicle_classifier/config.py`. Values passed directly to
the existing constructors take precedence over values in a config object.

`ModelConfig` owns checkpoint and runtime loading settings:

```python
from vehicle_classifier import ModelConfig, vehicle_classifier

model_config = ModelConfig(
  checkpoint_path="/path/to/best.pt",
  dinov2_repo_path="/path/to/dinov2",
  device="cuda",
  metadata_path="/path/to/metadata.json",
)
model = vehicle_classifier("/path/to/best.pt", model_config=model_config)
```

`ClassificationConfig` owns image preprocessing. Its defaults preserve the
original `224x224` aspect-preserving resize and padding behavior. Threshold and
smoothing fields are available for callers that add post-processing without
putting those concerns into model loading or costing.

`ExplicitFeatureConfig` owns penalty-matrix paths and all explicit-cost knobs:

```python
from vehicle_classifier import ExplicitFeatureConfig, ExplicitFeatureExtractor

explicit_config = ExplicitFeatureConfig(
  colour_matrix_path="/path/to/colour_final_penalty_matrix.csv",
  type_matrix_path="/path/to/type_final_penalty_matrix.csv",
  car_subtype_matrix_path="/path/to/car_subtype_final_penalty_matrix.csv",
  js_weight=0.5,
  feature_weights={"type": 0.65, "car_subtype": 0.20, "colour": 0.15},
  confidence_weighting=True,
)
extractor = ExplicitFeatureExtractor(
  model_config=model_config,
  explicit_config=explicit_config,
)
extractor.configure_cost_from_csv()
```

The equivalent legacy form remains valid and is useful when configuration is
assembled by the tracker at runtime:

Use `ExplicitFeatureExtractor` when you want one object that handles:

- crop/classifier embedding inference
- penalty-matrix configuration
- explicit feature cost scoring (single pair or matrix)

```python
from vehicle_classifier import ExplicitFeatureExtractor

extractor = ExplicitFeatureExtractor(
  checkpoints_path="/path/to/best.pt",
  dinov2_path="/path/to/dinov2",
)

extractor.configure_cost_from_csv(
  colour_matrix_path="/path/to/colour_final_penalty_matrix.csv",
  type_matrix_path="/path/to/type_final_penalty_matrix.csv",
  car_subtype_matrix_path="/path/to/car_subtype_final_penalty_matrix.csv",
  js_weight=0.5,
  feature_weights={"type": 0.65, "car_subtype": 0.20, "colour": 0.15},
)

# Single crop inference (expects BGR NumPy crop):
embedding_a = extractor.infer_embedding(crop_a)
embedding_b = extractor.infer_embedding(crop_b)

# Single-pair score:
cost = extractor.calculate_cost(embedding_a, embedding_b)

# Vectorised set score (NxM matrix):
cost_matrix = extractor.calculate_cost_matrix(
  [embedding_a, embedding_a2],
  [embedding_b, embedding_b2, embedding_b3],
)
```

`extract_embeddings(frame, detections)` remains available for tracker-style objects with `tlbr`, `embed_history`, and `embedding` fields.

## Example Runs

Use the generic loader when you do not want to care which training script produced the checkpoint:

```python
from vehicle_classifier import vehicle_classifier

model = vehicle_classifier(
  "/path/to/outputs/dinov2_vehicle_attrs/my_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
print(model.results("/path/to/crop.jpg"))
```

Checkpoint from `train_classifiers.py`:

```python
from vehicle_classifier import vehicle_classifier_base

model = vehicle_classifier_base(
  "/path/to/outputs/dinov2_vehicle_attrs/train_classifiers_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
print(result["colour"]["label"], result["type"]["label"], result["car_subtype"]["label"])
```

Checkpoint from `train_classifiers_with_size.py`:

```python
from vehicle_classifier import vehicle_classifier_with_size

model = vehicle_classifier_with_size(
  "/path/to/outputs/dinov2_vehicle_attrs/train_classifiers_with_size_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
print(result)
```

Checkpoint from `train_classifiers_intermediate.py`:

```python
from vehicle_classifier import vehicle_classifier_intermediate

model = vehicle_classifier_intermediate(
  "/path/to/outputs/dinov2_vehicle_attrs/train_classifiers_intermediate_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
print(result)
```

Checkpoint from `train_classifiers_intermediate_with_size.py`:

```python
from vehicle_classifier import vehicle_classifier_intermediate_with_size

model = vehicle_classifier_intermediate_with_size(
  "/path/to/outputs/dinov2_vehicle_attrs/train_classifiers_intermediate_with_size_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
print(result)
```

Checkpoint from type-only training:

```python
from vehicle_classifier import vehicle_classifier_type_only

model = vehicle_classifier_type_only(
  "/path/to/outputs/dinov2_vehicle_attrs/train_classifiers_intermediate_type_run/best.pt",
  dinov2_repo_path="/path/to/dinov2",
)
result = model.results("/path/to/vehicle_crop.jpg")
print(result["type"])
```

Equivalent shell one-liners:

```bash
python -c 'from vehicle_classifier import vehicle_classifier; m = vehicle_classifier("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_base; m = vehicle_classifier_base("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_with_size; m = vehicle_classifier_with_size("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_intermediate; m = vehicle_classifier_intermediate("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_intermediate_with_size; m = vehicle_classifier_intermediate_with_size("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_type_only; m = vehicle_classifier_type_only("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
python -c 'from vehicle_classifier import vehicle_classifier_type_only; m = vehicle_classifier_type_only("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2"); print(m.results("/path/to/crop.jpg"))'
```

`out` format:

```python
{
  "colour": {"index": 3, "label": "white", "confidence": 0.98},
  "type": {"index": 1, "label": "car", "confidence": 0.87},
  "car_subtype": {"index": 10, "label": "sedan", "confidence": 0.66},
  "feature_vector": [0.1, -0.4, ...],
  "checkpoint_variant": "intermediate_with_size",
}
```

`feature_vector` is the raw DINOv2 backbone descriptor for the crop (before any size fusion or intermediate-layer aggregation), independent of the classification heads above.

## Notes

- This package loads class names directly from the checkpoint.
- `dinov2_repo_path` must point to the local DINOv2 repo directory containing `hubconf.py`.
- The checkpoint must be produced by this project and include keys: `model`, `config`, and `type_classes`. `colour_classes` and `car_subtype_classes` are optional for type-only checkpoints.
- Inference preprocessing matches evaluation preprocessing from training: preserve aspect ratio, pad to `224x224`, and pass encoded original crop size to size-aware models.
- If trying to run on a normal image, object localisation must be done first, yolo for example. Then crop the image. Unfortunately, crop size/positioning has an effect on performance. Look at training images for examples, but often a buffer is needed (making larger crop than yolo box) for best performance. MIO is the best example (often quite zoomed out compared to close crops)
  - working on implementing yolo cropping atm, may incorporate into training process too, single shot process.
