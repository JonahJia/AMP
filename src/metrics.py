from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def upsample_map(heatmap: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Upsample a single 2D map to image resolution."""
    return F.interpolate(heatmap[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]


def normalize_map(heatmap: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    heatmap = heatmap - heatmap.min()
    return heatmap / (heatmap.max() + eps)


def heatmap_peak(heatmap: torch.Tensor) -> Tuple[int, int]:
    flat_idx = int(torch.argmax(heatmap).item())
    h, w = heatmap.shape
    y = flat_idx // w
    x = flat_idx % w
    return x, y


def load_mask_from_meta(meta: Dict[str, Any], target_size: Tuple[int, int]) -> torch.Tensor:
    """Load or synthesize an object mask aligned with the transformed image size.

    Priority:
      1) segmentation mask from meta['mask_path'] if available;
      2) bounding box from meta['bbox'] if available and crop_to_bbox is False;
      3) full-image mask as a fallback.
    """
    target_h, target_w = target_size
    mask_path = meta.get('mask_path')
    crop_bbox = meta.get('crop_bbox')
    crop_to_bbox = bool(meta.get('crop_to_bbox', False))
    bbox = meta.get('bbox')
    orig_w, orig_h = meta.get('original_size', (target_w, target_h))

    if mask_path is not None and Path(mask_path).exists():
        mask_img = Image.open(mask_path).convert('L')
        if crop_bbox is not None:
            mask_img = mask_img.crop(crop_bbox)
        mask_img = mask_img.resize((target_w, target_h), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0)
        return (mask > 0.5).float()

    mask = torch.zeros((target_h, target_w), dtype=torch.float32)

    if bbox is not None and not crop_to_bbox:
        x1, y1, x2, y2 = bbox
        sx = target_w / max(float(orig_w), 1.0)
        sy = target_h / max(float(orig_h), 1.0)
        x1 = int(round(x1 * sx))
        x2 = int(round(x2 * sx))
        y1 = int(round(y1 * sy))
        y2 = int(round(y2 * sy))
        x1 = max(0, min(x1, target_w - 1))
        x2 = max(x1 + 1, min(x2, target_w))
        y1 = max(0, min(y1, target_h - 1))
        y2 = max(y1 + 1, min(y2, target_h))
        mask[y1:y2, x1:x2] = 1.0
        return mask

    # Fallback: use the whole image as foreground.
    mask[:] = 1.0
    return mask


def transform_parts_to_eval_coords(meta: Dict[str, Any], target_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    parts = meta.get('parts') or []
    if not parts:
        return []
    crop_bbox = meta.get('crop_bbox')
    orig_w, orig_h = meta.get('original_size', target_size[::-1])
    target_h, target_w = target_size

    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        crop_w = max(x2 - x1, 1)
        crop_h = max(y2 - y1, 1)
        sx = target_w / crop_w
        sy = target_h / crop_h
    else:
        x1, y1 = 0, 0
        sx = target_w / max(float(orig_w), 1.0)
        sy = target_h / max(float(orig_h), 1.0)
        x2, y2 = orig_w, orig_h

    out = []
    for p in parts:
        if int(p.get('visible', 1)) <= 0:
            continue
        x = float(p['x']) - x1
        y = float(p['y']) - y1
        if x < 0 or y < 0 or x >= (x2 - x1) or y >= (y2 - y1):
            continue
        out.append({
            'name': str(p.get('name', p.get('label', 'part'))),
            'x': x * sx,
            'y': y * sy,
        })
    return out


def assign_region_label(meta: Dict[str, Any], peak_xy: Tuple[int, int], target_size: Tuple[int, int], grid_size: int = 3) -> str:
    x, y = peak_xy
    parts = transform_parts_to_eval_coords(meta, target_size)
    if parts:
        best_name = None
        best_dist = None
        for p in parts:
            dist = (p['x'] - x) ** 2 + (p['y'] - y) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = p['name']
        return str(best_name)

    target_h, target_w = target_size
    bbox = meta.get('bbox')
    crop_to_bbox = bool(meta.get('crop_to_bbox', False))
    orig_w, orig_h = meta.get('original_size', (target_w, target_h))
    if bbox is not None and not crop_to_bbox:
        x1, y1, x2, y2 = bbox
        sx = target_w / max(float(orig_w), 1.0)
        sy = target_h / max(float(orig_h), 1.0)
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(target_w), float(target_h)

    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    rel_x = min(max((x - x1) / box_w, 0.0), 0.999999)
    rel_y = min(max((y - y1) / box_h, 0.0), 0.999999)
    gx = int(rel_x * grid_size)
    gy = int(rel_y * grid_size)
    return f'grid_{gy}_{gx}'


@torch.no_grad()
def compute_dauc_for_sample(
    model,
    image: torch.Tensor,
    heatmap: torch.Tensor,
    pred_class: int,
    steps: int = 20,
    deletion_batch_size: int = 8,
) -> float:
    """Deletion AUC on a single normalized image tensor."""
    if image.ndim != 3:
        raise ValueError('image must have shape (C, H, W)')
    h, w = heatmap.shape
    scores = heatmap.flatten()
    ranking = torch.argsort(scores, descending=True)
    num_pixels = h * w
    images = []
    for t in range(1, steps + 1):
        frac = t / steps
        k = max(1, int(round(frac * num_pixels)))
        masked = image.clone().view(image.size(0), -1)
        masked[:, ranking[:k]] = 0.0
        images.append(masked.view_as(image))

    all_scores: List[torch.Tensor] = []
    for start in range(0, len(images), deletion_batch_size):
        batch = torch.stack(images[start:start + deletion_batch_size], dim=0).to(next(model.parameters()).device)
        logits = model(batch).logits
        probs = torch.softmax(logits, dim=1)[:, pred_class]
        all_scores.append(probs.detach().cpu())
    return torch.cat(all_scores).mean().item()


@torch.no_grad()
def evaluate_interpretability(model, dataloader, eval_config, device: str = 'cuda') -> Dict[str, float]:
    """Compute accuracy + explanation metrics.

    The implementation is intentionally generic: if segmentation masks or keypoint
    annotations are available, they will be used; otherwise the evaluation falls
    back to canonical-grid regions and bbox/full-image masks.
    """
    model.eval()
    num_total = 0
    num_correct = 0

    oirr_vals: List[float] = []
    dauc_vals: List[float] = []
    pg_vals: List[float] = []
    bg_vals: List[float] = []

    # class -> direction -> labels
    assignments: DefaultDict[int, DefaultDict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    stability_hits: List[float] = []

    for images, labels, metas in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        forward = model(images)
        explain = model.explain(images)
        preds = explain['pred']
        num_total += labels.size(0)
        num_correct += (preds == labels).sum().item()

        target_size = (images.size(-2), images.size(-1))
        noisy_images = images + eval_config.perturb_sigma * torch.randn_like(images)
        noisy_forward = model(noisy_images)

        active_mask_all = model.get_active_mask().detach().cpu()

        for i in range(images.size(0)):
            pred = int(preds[i].item())
            label = int(labels[i].item())
            if eval_config.correct_only and pred != label:
                continue

            agg = upsample_map(explain['aggregate_heatmaps'][i].cpu(), target_size)
            agg = normalize_map(agg)
            mask = load_mask_from_meta(metas[i], target_size)
            inside_mean = float(agg[mask > 0.5].mean().item()) if (mask > 0.5).any() else 0.0
            outside_mean = float(agg[mask <= 0.5].mean().item()) if (mask <= 0.5).any() else 0.0
            if inside_mean > 0:
                oirr_vals.append(outside_mean / inside_mean)
            else:
                oirr_vals.append(0.0)
            bg_vals.append(float(agg[mask <= 0.5].sum().item() / (agg.sum().item() + 1e-6)))
            peak_x, peak_y = heatmap_peak(agg)
            pg_vals.append(float(mask[peak_y, peak_x].item() > 0.5))
            dauc_vals.append(
                compute_dauc_for_sample(
                    model,
                    images[i].detach().cpu(),
                    agg.detach().cpu(),
                    pred_class=pred,
                    steps=eval_config.dauc_steps,
                    deletion_batch_size=eval_config.deletion_batch_size,
                )
            )

            active_dirs = active_mask_all[pred]
            for k, is_active in enumerate(active_dirs.tolist()):
                if not is_active:
                    continue
                dir_map = forward.weighted_maps[i, pred, k].detach().cpu()
                dir_map = normalize_map(upsample_map(dir_map, target_size))
                label_name = assign_region_label(metas[i], heatmap_peak(dir_map), target_size, eval_config.grid_size)
                assignments[pred][k].append(label_name)

                noisy_dir_map = noisy_forward.weighted_maps[i, pred, k].detach().cpu()
                noisy_dir_map = normalize_map(upsample_map(noisy_dir_map, target_size))
                noisy_label = assign_region_label(metas[i], heatmap_peak(noisy_dir_map), target_size, eval_config.grid_size)
                stability_hits.append(float(label_name == noisy_label))

    consistency_total = 0
    consistency_hits = 0
    for _, direction_dict in assignments.items():
        for _, labels_list in direction_dict.items():
            if len(labels_list) == 0:
                continue
            counts = Counter(labels_list)
            dominant = max(counts.values()) / len(labels_list)
            consistency_total += 1
            consistency_hits += float(dominant >= eval_config.consistency_threshold)

    return {
        'accuracy': 100.0 * num_correct / max(num_total, 1),
        'consistency': 100.0 * consistency_hits / max(consistency_total, 1),
        'stability': 100.0 * (sum(stability_hits) / max(len(stability_hits), 1)),
        'oirr': float(np.mean(oirr_vals)) if oirr_vals else 0.0,
        'dauc': float(np.mean(dauc_vals)) if dauc_vals else 0.0,
        'pg': 100.0 * float(np.mean(pg_vals)) if pg_vals else 0.0,
        'bg_leak': float(np.mean(bg_vals)) if bg_vals else 0.0,
    }
