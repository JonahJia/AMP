from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .receptive_field import compute_rf_prototype


@torch.no_grad()
def build_exemplar_cache(
    model,
    dataloader,
    device: str = 'cuda',
    top_m: int = 5,
) -> Dict[str, Any]:
    """Build class- and direction-specific exemplar cache.

    For each class-specific direction k, the cache stores the top-m training images
    and spatial locations that maximize (u_{c,k}^T f_{h,w})^2 over the training set.
    """
    model.eval()
    cache: DefaultDict[int, DefaultDict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    active_mask_all = model.get_active_mask().detach().cpu()

    for images, labels, metas in dataloader:
        images = images.to(device)
        out = model(images)
        basis_maps = out.basis_maps.detach().cpu()  # (B, C, K, H, W)

        for i in range(images.size(0)):
            c = int(labels[i].item())
            rel_path = metas[i].get('rel_path', Path(metas[i]['path']).name)
            for k, is_active in enumerate(active_mask_all[c].tolist()):
                if not is_active:
                    continue
                score_map = basis_maps[i, c, k]
                flat_idx = int(torch.argmax(score_map).item())
                h = flat_idx // score_map.size(1)
                w = flat_idx % score_map.size(1)
                score = float(score_map[h, w].item())
                if model.proto_layer_rf_info is not None:
                    _, rf_h_start, rf_h_end, rf_w_start, rf_w_end = compute_rf_prototype(
                        model.img_size,
                        (0, h, w),
                        model.proto_layer_rf_info,
                    )
                    rf_box = [rf_h_start, rf_h_end, rf_w_start, rf_w_end]
                else:
                    rf_box = None
                entry = {
                    'path': metas[i]['path'],
                    'rel_path': rel_path,
                    'score': score,
                    'feature_hw': [int(h), int(w)],
                    'rf_box': rf_box,
                }
                bucket = cache[c][k]
                bucket.append(entry)
                bucket.sort(key=lambda x: x['score'], reverse=True)
                if len(bucket) > top_m:
                    del bucket[top_m:]

    return {
        'top_m': top_m,
        'cache': {int(c): {int(k): v for k, v in dir_dict.items()} for c, dir_dict in cache.items()},
    }

def save_exemplar_cache(cache: Dict[str, Any], path: str | Path) -> None:
    torch.save(cache, Path(path))

def load_exemplar_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location='cpu')

def _effective_rank_from_matrix(a: torch.Tensor) -> float:
    s = torch.linalg.svdvals(a)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).item())

@torch.no_grad()
def compute_collapse_diagnostics(
    model,
    dataloader,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Compute pairwise cosine, effective rank, leading singular mass,
    heatmap overlap, and an NC-style covariance ratio.
    """
    model.eval()
    bases = model.class_bases.detach().cpu()  # (C, D, K)
    caps = model.capacities.detach().cpu().clamp_min(0.0)
    active_mask = (caps > model.active_threshold)

    pair_cos_vals: List[float] = []
    erank_vals: List[float] = []
    rho1_vals: List[float] = []

    for c in range(model.num_classes):
        uc = bases[c]  # (D, K)
        # Pairwise cosine on basis columns.
        normed = F.normalize(uc.t(), dim=1)
        sim = normed @ normed.t()
        k = sim.size(0)
        if k > 1:
            pair_cos = (sim.abs().sum() - torch.trace(sim.abs())) / (k * (k - 1))
            pair_cos_vals.append(float(pair_cos.item()))
        ac = uc * caps[c].unsqueeze(0)
        erank_vals.append(_effective_rank_from_matrix(ac))
        s = torch.linalg.svdvals(ac)
        if s.sum() > 0:
            rho1_vals.append(float((s[0] / s.sum()).item()))
        else:
            rho1_vals.append(0.0)

    heatmap_overlaps: List[float] = []
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch_idx, (images, labels, metas) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        preds = out.logits.argmax(dim=1)
        weighted = out.weighted_maps.detach().cpu()

        # Collect spatial features for NC ratio.
        feats = out.features.detach().cpu().permute(0, 2, 3, 1).reshape(-1, out.features.size(1))
        repeated_labels = labels.cpu().repeat_interleave(out.features.size(2) * out.features.size(3))
        all_feats.append(feats)
        all_labels.append(repeated_labels)

        for i in range(images.size(0)):
            c = int(preds[i].item())
            if preds[i] != labels[i]:
                continue
            active = active_mask[c]
            if int(active.sum().item()) < 2:
                continue
            maps = weighted[i, c, active]  # (R, H, W)
            flat = maps.flatten(1)
            normed = F.normalize(flat, dim=1)
            sim = normed @ normed.t()
            r = sim.size(0)
            overlap = (sim.sum() - torch.trace(sim)) / (r * (r - 1))
            heatmap_overlaps.append(float(overlap.item()))

    if all_feats:
        feats = torch.cat(all_feats, dim=0)
        labels = torch.cat(all_labels, dim=0)
        overall_mean = feats.mean(dim=0)
        unique_labels = labels.unique(sorted=True)
        within_trace = 0.0
        between_trace = 0.0
        for c in unique_labels.tolist():
            cls_feats = feats[labels == c]
            if cls_feats.numel() == 0:
                continue
            mean_c = cls_feats.mean(dim=0)
            centered = cls_feats - mean_c
            within_trace += float((centered.square().sum() / max(cls_feats.size(0), 1)).item())
            between_trace += float((mean_c - overall_mean).square().sum().item())
        nc_ratio = within_trace / max(between_trace, 1e-8)
    else:
        nc_ratio = 0.0

    return {
        'pair_cosine': float(np.mean(pair_cos_vals)) if pair_cos_vals else 0.0,
        'effective_rank': float(np.mean(erank_vals)) if erank_vals else 0.0,
        'rho1': float(np.mean(rho1_vals)) if rho1_vals else 0.0,
        'heatmap_overlap': float(np.mean(heatmap_overlaps)) if heatmap_overlaps else 0.0,
        'nc_ratio': nc_ratio,
        'avg_active_rank': float(active_mask.sum(dim=1).float().mean().item()),
    }
