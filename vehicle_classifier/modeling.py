from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Mapping
from typing import cast

import torch
import torch.nn as nn


FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vitg14_reg": 1536,
}

INTERMEDIATE_BLOCKS = [5, 8, 11]
SIZE_AWARE_VARIANTS = {
    "with_size",
    "intermediate",
    "intermediate_with_size",
    "type_only_with_size",
    "type_only_intermediate",
    "type_only_intermediate_with_size",
}


def _feature_dim_for_backbone(backbone_name: str) -> int:
    if backbone_name not in FEATURE_DIMS:
        raise ValueError(
            f"Unknown backbone {backbone_name!r}. Known options: {sorted(FEATURE_DIMS)}"
        )
    return FEATURE_DIMS[backbone_name]


def _load_backbone(backbone_name: str, dinov2_repo_path: str | Path) -> nn.Module:
    repo_path = Path(dinov2_repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv2 repo path not found: {repo_path}")

    load_kwargs: dict[str, Any] = {
        "source": "local",
        "pretrained": False,
    }
    return cast(nn.Module, torch.hub.load(str(repo_path), backbone_name, **load_kwargs))


def _make_head(feat_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feat_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, out_dim),
    )


def infer_model_variant(state_dict: Mapping[str, torch.Tensor]) -> str:
    has_size_projection = "size_projection.0.weight" in state_dict
    has_aggregation = "aggregation_mlp.0.weight" in state_dict
    has_head_norm = "head_norm.weight" in state_dict
    has_norm = "norm.weight" in state_dict
    has_colour_head = "colour_head.0.weight" in state_dict
    has_car_subtype_head = "car_subtype_head.0.weight" in state_dict
    has_type_head = "type_head.0.weight" in state_dict

    if not has_type_head:
        raise ValueError("Checkpoint does not contain a type head; unsupported checkpoint format")

    is_type_only = not has_colour_head and not has_car_subtype_head

    if has_aggregation:
        if not has_size_projection or not has_head_norm:
            raise ValueError("Checkpoint looks like an intermediate model but is missing required layers")

        head_input_dim = int(state_dict["type_head.0.weight"].shape[1])
        aggregated_dim = int(state_dict["aggregation_mlp.3.weight"].shape[0])
        if head_input_dim == aggregated_dim:
            return "type_only_intermediate" if is_type_only else "intermediate"
        if head_input_dim > aggregated_dim:
            return "type_only_intermediate_with_size" if is_type_only else "intermediate_with_size"
        raise ValueError("Checkpoint head dimensions do not match a known intermediate variant")

    if has_size_projection:
        if not has_norm:
            raise ValueError("Checkpoint looks size-aware but is missing the expected layer norm")
        return "type_only_with_size" if is_type_only else "with_size"

    if has_norm:
        return "type_only_baseline" if is_type_only else "baseline"

    raise ValueError("Unable to infer checkpoint architecture variant from state_dict keys")


def variant_uses_image_size(variant: str) -> bool:
    return variant in SIZE_AWARE_VARIANTS


def build_model_for_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    *,
    backbone_name: str,
    n_colours: int,
    n_types: int,
    n_subtypes: int,
    dinov2_repo_path: str | Path = "dinov2",
) -> tuple[nn.Module, str]:
    variant = infer_model_variant(state_dict)
    if "colour_head.0.weight" in state_dict:
        hidden_dim = int(state_dict["colour_head.0.weight"].shape[0])
    else:
        hidden_dim = int(state_dict["type_head.0.weight"].shape[0])

    if variant == "baseline":
        model = DinoVehicleAttributeModel(
            backbone_name=backbone_name,
            n_colours=n_colours,
            n_types=n_types,
            n_subtypes=n_subtypes,
            hidden_dim=hidden_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "with_size":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleAttributeModelWithSize(
            backbone_name=backbone_name,
            n_colours=n_colours,
            n_types=n_types,
            n_subtypes=n_subtypes,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "intermediate":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleAttributeModelIntermediate(
            backbone_name=backbone_name,
            n_colours=n_colours,
            n_types=n_types,
            n_subtypes=n_subtypes,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "intermediate_with_size":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleAttributeModelIntermediateWithSize(
            backbone_name=backbone_name,
            n_colours=n_colours,
            n_types=n_types,
            n_subtypes=n_subtypes,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "type_only_baseline":
        model = DinoVehicleTypeOnlyModel(
            backbone_name=backbone_name,
            n_types=n_types,
            hidden_dim=hidden_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "type_only_with_size":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleTypeOnlyModelWithSize(
            backbone_name=backbone_name,
            n_types=n_types,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "type_only_intermediate":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleTypeOnlyModelIntermediate(
            backbone_name=backbone_name,
            n_types=n_types,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    elif variant == "type_only_intermediate_with_size":
        size_emb_dim = int(state_dict["size_projection.0.weight"].shape[0])
        model = DinoVehicleTypeOnlyModelIntermediateWithSize(
            backbone_name=backbone_name,
            n_types=n_types,
            hidden_dim=hidden_dim,
            size_emb_dim=size_emb_dim,
            dinov2_repo_path=dinov2_repo_path,
        )
    else:
        raise ValueError(f"Unsupported checkpoint variant: {variant}")

    return model, variant


class DinoVehicleAttributeModel(nn.Module):
    """DINOv2 backbone plus three supervised heads."""

    def __init__(
        self,
        backbone_name: str,
        n_colours: int,
        n_types: int,
        n_subtypes: int,
        hidden_dim: int = 512,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.norm = nn.LayerNorm(feat_dim)
        self.colour_head = _make_head(feat_dim, hidden_dim, n_colours)
        self.type_head = _make_head(feat_dim, hidden_dim, n_types)
        self.car_subtype_head = _make_head(feat_dim, hidden_dim, n_subtypes)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.norm(features)
        return {
            "colour": self.colour_head(features),
            "type": self.type_head(features),
            "car_subtype": self.car_subtype_head(features),
        }


class DinoVehicleAttributeModelWithSize(nn.Module):
    """DINOv2 backbone plus three supervised heads conditioned on image size."""

    def __init__(
        self,
        backbone_name: str,
        n_colours: int,
        n_types: int,
        n_subtypes: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        head_input_dim = feat_dim + size_emb_dim

        self.norm = nn.LayerNorm(head_input_dim)
        self.colour_head = _make_head(head_input_dim, hidden_dim, n_colours)
        self.type_head = _make_head(head_input_dim, hidden_dim, n_types)
        self.car_subtype_head = _make_head(head_input_dim, hidden_dim, n_subtypes)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        size_features = self.size_projection(image_size.float())
        features = torch.cat([features, size_features], dim=1)
        features = self.norm(features)
        return {
            "colour": self.colour_head(features),
            "type": self.type_head(features),
            "car_subtype": self.car_subtype_head(features),
        }


class DinoVehicleTypeOnlyModel(nn.Module):
    """DINOv2 backbone plus a single type head."""

    def __init__(
        self,
        backbone_name: str,
        n_types: int,
        hidden_dim: int = 512,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.norm = nn.LayerNorm(feat_dim)
        self.type_head = _make_head(feat_dim, hidden_dim, n_types)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.norm(features)
        return {"type": self.type_head(features)}


class DinoVehicleTypeOnlyModelWithSize(nn.Module):
    """DINOv2 backbone plus a single type head conditioned on image size."""

    def __init__(
        self,
        backbone_name: str,
        n_types: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        self.norm = nn.LayerNorm(feat_dim + size_emb_dim)
        self.type_head = _make_head(feat_dim + size_emb_dim, hidden_dim, n_types)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        size_features = self.size_projection(image_size.float())
        features = torch.cat([features, size_features], dim=1)
        features = self.norm(features)
        return {"type": self.type_head(features)}


class _IntermediateBackboneMixin:
    def _extract_block_embedding(self, layer_output: Any) -> torch.Tensor:
        if isinstance(layer_output, (tuple, list)):
            layer_parts = list(layer_output)
            if not layer_parts:
                raise ValueError("Expected a non-empty intermediate layer output")
            token = layer_parts[1] if len(layer_parts) > 1 else layer_parts[0]
        else:
            token = layer_output

        if token.dim() == 3:
            return token[:, 0]
        return token

    def _aggregate_intermediates(
        self,
        images: torch.Tensor,
        image_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_layers = self.backbone.get_intermediate_layers(
            images,
            n=INTERMEDIATE_BLOCKS,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        size_features = self.size_projection(image_size.float())

        block_embeddings = []
        for layer_output in raw_layers:
            block_embeddings.append(self._extract_block_embedding(layer_output))

        aggregation_input = torch.cat([torch.cat(block_embeddings, dim=1), size_features], dim=1)
        aggregated = self.aggregation_mlp(aggregation_input)
        return aggregated, size_features


class DinoVehicleTypeOnlyModelIntermediate(_IntermediateBackboneMixin, nn.Module):
    """DINOv2 backbone with intermediate layer aggregation and a single type head."""

    def __init__(
        self,
        backbone_name: str,
        n_types: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(len(INTERMEDIATE_BLOCKS) * feat_dim + size_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.head_norm = nn.LayerNorm(feat_dim)
        self.type_head = _make_head(feat_dim, hidden_dim, n_types)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        aggregated_features, _size_features = self._aggregate_intermediates(images, image_size)
        head_features = self.head_norm(aggregated_features)
        return {"type": self.type_head(head_features)}


class DinoVehicleTypeOnlyModelIntermediateWithSize(_IntermediateBackboneMixin, nn.Module):
    """DINOv2 backbone with intermediate aggregation plus size-aware single type head."""

    def __init__(
        self,
        backbone_name: str,
        n_types: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(len(INTERMEDIATE_BLOCKS) * feat_dim + size_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
        )

        head_input_dim = feat_dim + size_emb_dim
        self.head_norm = nn.LayerNorm(head_input_dim)
        self.type_head = _make_head(head_input_dim, hidden_dim, n_types)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        aggregated_features, size_features = self._aggregate_intermediates(images, image_size)
        head_features = torch.cat([aggregated_features, size_features], dim=1)
        head_features = self.head_norm(head_features)
        return {"type": self.type_head(head_features)}


class DinoVehicleAttributeModelIntermediate(_IntermediateBackboneMixin, nn.Module):
    """DINOv2 backbone with size-conditioned intermediate-layer aggregation."""

    def __init__(
        self,
        backbone_name: str,
        n_colours: int,
        n_types: int,
        n_subtypes: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(len(INTERMEDIATE_BLOCKS) * feat_dim + size_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.head_norm = nn.LayerNorm(feat_dim)
        self.colour_head = _make_head(feat_dim, hidden_dim, n_colours)
        self.type_head = _make_head(feat_dim, hidden_dim, n_types)
        self.car_subtype_head = _make_head(feat_dim, hidden_dim, n_subtypes)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        aggregated_features, _size_features = self._aggregate_intermediates(images, image_size)
        head_features = self.head_norm(aggregated_features)
        return {
            "colour": self.colour_head(head_features),
            "type": self.type_head(head_features),
            "car_subtype": self.car_subtype_head(head_features),
        }


class DinoVehicleAttributeModelIntermediateWithSize(_IntermediateBackboneMixin, nn.Module):
    """DINOv2 backbone with intermediate aggregation and size-aware heads."""

    def __init__(
        self,
        backbone_name: str,
        n_colours: int,
        n_types: int,
        n_subtypes: int,
        hidden_dim: int = 512,
        size_emb_dim: int = 32,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        feat_dim = _feature_dim_for_backbone(backbone_name)
        self.backbone = _load_backbone(backbone_name, dinov2_repo_path)

        self.size_projection = nn.Sequential(
            nn.Linear(2, size_emb_dim),
            nn.GELU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(len(INTERMEDIATE_BLOCKS) * feat_dim + size_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
        )

        head_input_dim = feat_dim + size_emb_dim
        self.head_norm = nn.LayerNorm(head_input_dim)
        self.colour_head = _make_head(head_input_dim, hidden_dim, n_colours)
        self.type_head = _make_head(head_input_dim, hidden_dim, n_types)
        self.car_subtype_head = _make_head(head_input_dim, hidden_dim, n_subtypes)

    def forward(self, images: torch.Tensor, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
        aggregated_features, size_features = self._aggregate_intermediates(images, image_size)
        head_features = torch.cat([aggregated_features, size_features], dim=1)
        head_features = self.head_norm(head_features)
        return {
            "colour": self.colour_head(head_features),
            "type": self.type_head(head_features),
            "car_subtype": self.car_subtype_head(head_features),
        }