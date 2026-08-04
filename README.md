# MOT_classifier

Inference package for checkpoints saved by this repository's DINOv2 trainers:

- `train_classifiers.py`
- `train_classifiers_with_size.py`
- `train_classifiers_intermediate.py`
- `train_classifiers_intermediate_with_size.py`
- `train_classifiers_intermediate_type.py`

## Install

From this repository root:

```bash
pip install -e MOT_classifier
```

## Usage

```python
from vehicle_classifier import vehicle_classifier

model = vehicle_classifier("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
out = model.results("/path/to/image.jpg")
print(out)
```

`vehicle_classifier(...)` auto-detects which checkpoint architecture it is loading.

Explicit entry points for each experimental trainer are also available:

```python
from vehicle_classifier import vehicle_classifier_base
from vehicle_classifier import vehicle_classifier_with_size
from vehicle_classifier import vehicle_classifier_intermediate
from vehicle_classifier import vehicle_classifier_intermediate_with_size
from vehicle_classifier import vehicle_classifier_type_only

base_model = vehicle_classifier_base("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
size_model = vehicle_classifier_with_size("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
intermediate_model = vehicle_classifier_intermediate("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
intermediate_size_model = vehicle_classifier_intermediate_with_size("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
type_only_model = vehicle_classifier_type_only("/path/to/best.pt", dinov2_repo_path="/path/to/dinov2")
```

## Explicit Extractor API

Use `ExplicitExtractor` when you want one object that handles:

- crop/classifier embedding inference
- penalty-matrix configuration
- explicit feature cost scoring (single pair or matrix)

```python
from vehicle_classifier import ExplicitExtractor

extractor = ExplicitExtractor(
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
  "checkpoint_variant": "intermediate_with_size",
}
```

## Notes

- This package loads class names directly from the checkpoint.
- `dinov2_repo_path` must point to the local DINOv2 repo directory containing `hubconf.py`.
- The checkpoint must be produced by this project and include keys: `model`, `config`, and `type_classes`. `colour_classes` and `car_subtype_classes` are optional for type-only checkpoints.
- Inference preprocessing matches evaluation preprocessing from training: preserve aspect ratio, pad to `224x224`, and pass encoded original crop size to size-aware models.
- If trying to run on a normal image, object localisation must be done first, yolo for example. Then crop the image. Unfortunately, crop size/positioning has an effect on performance. Look at training images for examples, but often a buffer is needed (making larger crop than yolo box) for best performance. MIO is the best example (often quite zoomed out compared to close crops)
  - working on implementing yolo cropping atm, may incorporate into training process too, single shot process.
