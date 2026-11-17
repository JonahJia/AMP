from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone
from .receptive_field import compute_proto_layer_rf_info


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    basis_maps: torch.Tensor
    weighted_maps: torch.Tensor
    max_scores: torch.Tensor
    capacities: torch.Tensor


class AdaptiveManifoldPrototypes(nn.Module):
    """Class-specific Stiefel bases with sparse nonnegative capacities."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        prototype_dim: int,
        subspace_dim: int,
        img_size: int = 448,
        active_threshold: float = 1e-6,
        explain_threshold: float = 1e-4,
        add_on_layers_type: str = 'regular',
        eps: float = 1e-6,
        proto_layer_rf_info: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.features = backbone
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.subspace_dim = subspace_dim
        self.img_size = img_size
        self.active_threshold = active_threshold
        self.explain_threshold = explain_threshold
        self.eps = eps
        self.proto_layer_rf_info = proto_layer_rf_info

        backbone_out_channels = self._get_backbone_output_channels()
        self.add_on_layers = self._build_add_on_layers(
            backbone_out_channels,
            prototype_dim,
            add_on_layers_type,
        )

        # U: (C, D, K)
        self.class_bases = nn.Parameter(
            torch.empty(num_classes, prototype_dim, subspace_dim)
        )
        # sigma: (C, K)
        self.capacities = nn.Parameter(torch.ones(num_classes, subspace_dim))

        self._initialize_parameters()

    def _get_backbone_output_channels(self) -> int:
        for module in reversed(list(self.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
            if isinstance(module, nn.BatchNorm2d):
                return module.num_features
        raise ValueError('Could not determine backbone output channels')

    def _build_add_on_layers(self, in_channels: int, out_channels: int, layer_type: str) -> nn.Sequential:
        if layer_type == 'bottleneck':
            layers: List[nn.Module] = []
            current = in_channels
            while current > out_channels or len(layers) == 0:
                nxt = max(out_channels, current // 2)
                layers.extend([
                    nn.Conv2d(current, nxt, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nxt, nxt, kernel_size=1),
                ])
                if nxt > out_channels:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.Sigmoid())
                current = nxt
            return nn.Sequential(*layers)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _initialize_parameters(self) -> None:
        # Add-on layers
        for module in self.add_on_layers.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Orthonormalize each class basis independently.
        with torch.no_grad():
            for c in range(self.num_classes):
                q, r = torch.linalg.qr(torch.randn(self.prototype_dim, self.subspace_dim), mode='reduced')
                signs = torch.sign(torch.diag(r))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                q = q * signs.unsqueeze(0)
                self.class_bases[c].copy_(q)
            self.capacities.fill_(1.0)

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def get_active_mask(self, threshold: Optional[float] = None) -> torch.Tensor:
        thr = self.active_threshold if threshold is None else threshold
        return (self.capacities > thr)

    def _compute_basis_maps(self, features: torch.Tensor) -> torch.Tensor:
        """Return squared projections with shape (B, C, K, H, W)."""
        b, d, h, w = features.shape
        feats = features.view(b, d, h * w)
        proj = torch.einsum('bdn,cdk->bckn', feats, self.class_bases)
        basis_maps = proj.square().view(b, self.num_classes, self.subspace_dim, h, w)
        return basis_maps

    def _compute_weighted_maps(self, basis_maps: torch.Tensor) -> torch.Tensor:
        capacities = self.capacities.clamp_min(0.0)
        return basis_maps * capacities[None, :, :, None, None]

    def forward(self, x: torch.Tensor) -> ModelOutput:
        features = self.conv_features(x)
        basis_maps = self._compute_basis_maps(features)
        weighted_maps = self._compute_weighted_maps(basis_maps)
        max_scores = weighted_maps.amax(dim=-1).amax(dim=-1)  # (B, C, K)
        logits = max_scores.sum(dim=-1)
        return ModelOutput(
            logits=logits,
            features=features,
            basis_maps=basis_maps,
            weighted_maps=weighted_maps,
            max_scores=max_scores,
            capacities=self.capacities.clamp_min(0.0),
        )

    def compute_spatial_regularizers(self, basis_maps: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute L_SEM and L_overlap on the active set of the ground-truth class."""
        batch_sem: List[torch.Tensor] = []
        batch_overlap: List[torch.Tensor] = []
        device = basis_maps.device
        active_mask = self.get_active_mask().detach()

        for i in range(labels.size(0)):
            c = int(labels[i].item())
            maps = basis_maps[i, c]  # (K, H, W), unweighted as in the paper
            mask = active_mask[c]
            if not mask.any():
                batch_sem.append(torch.zeros((), device=device))
                batch_overlap.append(torch.zeros((), device=device))
                continue
            active_maps = maps[mask]  # (R, H, W)
            flat = active_maps.flatten(1)
            probs = torch.softmax(flat, dim=-1)
            sem = -(probs * torch.log(probs + self.eps)).sum(dim=-1).mean()
            batch_sem.append(sem)

            if probs.size(0) < 2:
                batch_overlap.append(torch.zeros((), device=device))
            else:
                normed = F.normalize(probs, dim=-1)
                sim = normed @ normed.t()
                r = sim.size(0)
                overlap = (sim.sum() - torch.trace(sim)) / (r * (r - 1))
                batch_overlap.append(overlap)

        return {
            'sem': torch.stack(batch_sem).mean() if batch_sem else torch.zeros((), device=device),
            'overlap': torch.stack(batch_overlap).mean() if batch_overlap else torch.zeros((), device=device),
        }

    @torch.no_grad()
    def explain(
        self,
        x: torch.Tensor,
        explain_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        pred = out.logits.argmax(dim=1)
        thr = self.explain_threshold if explain_threshold is None else explain_threshold
        batch_size = x.size(0)
        agg_maps: List[torch.Tensor] = []
        visible_masks: List[torch.Tensor] = []
        per_dir_maps: List[torch.Tensor] = []
        per_dir_scores: List[torch.Tensor] = []

        for i in range(batch_size):
            c = int(pred[i].item())
            maps = out.weighted_maps[i, c]  # (K, H, W)
            scores = maps.amax(dim=-1).amax(dim=-1)
            visible = scores > thr
            if not visible.any():
                visible = scores == scores.max()
            vis_maps = maps[visible]
            agg = vis_maps.sum(dim=0)
            agg = agg - agg.min()
            agg = agg / (agg.max() + self.eps)
            agg_maps.append(agg)
            visible_masks.append(visible)
            per_dir_maps.append(maps)
            per_dir_scores.append(scores)

        return {
            'pred': pred,
            'aggregate_heatmaps': torch.stack(agg_maps),
            'visible_masks': torch.stack(visible_masks),
            'per_direction_maps': torch.stack(per_dir_maps),
            'per_direction_scores': torch.stack(per_dir_scores),
            'logits': out.logits,
            'features': out.features,
        }

    @property
    def prototype_shape(self) -> Tuple[int, int, int]:
        return self.num_classes, self.prototype_dim, self.subspace_dim



def build_model(
    base_architecture: str = 'resnet50',
    pretrained: bool = True,
    num_classes: int = 200,
    prototype_dim: int = 128,
    subspace_dim: int = 10,
    img_size: int = 448,
    active_threshold: float = 1e-6,
    explain_threshold: float = 1e-4,
    add_on_layers_type: str = 'regular',
    eps: float = 1e-6,
) -> AdaptiveManifoldPrototypes:
    backbone, backbone_info = get_backbone(base_architecture, pretrained=pretrained)
    proto_layer_rf_info = compute_proto_layer_rf_info(
        img_size=img_size,
        layer_filter_sizes=backbone_info['filter_sizes'],
        layer_strides=backbone_info['strides'],
        layer_paddings=backbone_info['paddings'],
        prototype_kernel_size=1,
    )
    return AdaptiveManifoldPrototypes(
        backbone=backbone,
        num_classes=num_classes,
        prototype_dim=prototype_dim,
        subspace_dim=subspace_dim,
        img_size=img_size,
        active_threshold=active_threshold,
        explain_threshold=explain_threshold,
        add_on_layers_type=add_on_layers_type,
        eps=eps,
        proto_layer_rf_info=proto_layer_rf_info,
    )
