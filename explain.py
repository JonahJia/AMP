from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import utils.torchvision_compat  # noqa: F401
from torchvision import datasets, transforms
from configs.config import Config
from src.analysis import load_exemplar_cache
from src.model import build_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate AMP explanations for images')
    parser.add_argument('--config', type=str, required=True, help='AMP YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='AMP checkpoint')
    parser.add_argument('--cache', type=str, default=None, help='Exemplar cache (.pt)')
    parser.add_argument('--images', nargs='*', default=None, help='Image paths to explain')
    parser.add_argument('--output-dir', type=str, default='explanations', help='Output directory')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--top-m', type=int, default=3, help='Number of exemplars per visible direction')
    return parser.parse_args()

def _denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return t.permute(1, 2, 0).cpu().numpy()

def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap('jet')
    color = cmap(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1 - alpha) * image + alpha * color, 0, 1)

def _load_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def _load_class_names(config: Config) -> List[str]:
    dataset = datasets.ImageFolder(config.data.train_path)
    return dataset.classes

def _load_exemplar_patch(entry: dict) -> Image.Image:
    path = Path(entry['path'])
    img = Image.open(path).convert('RGB')
    rf_box = entry.get('rf_box')
    if rf_box is not None:
        y1, y2, x1, x2 = rf_box
        img = img.crop((x1, y1, x2, y2))
    return img

def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    config = Config.from_yaml(args.config)
    device = config.device if torch.cuda.is_available() and config.device.startswith('cuda') else 'cpu'
    class_names = _load_class_names(config)
    transform = _load_eval_transform(config.model.img_size)

    model = build_model(
        base_architecture=config.model.base_architecture,
        pretrained=False,
        num_classes=config.model.num_classes,
        prototype_dim=config.model.prototype_dim,
        subspace_dim=config.model.subspace_dim,
        img_size=config.model.img_size,
        active_threshold=config.model.active_threshold,
        explain_threshold=config.model.explain_threshold,
        add_on_layers_type=config.model.add_on_layers_type,
        eps=config.model.eps,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    cache = load_exemplar_cache(args.cache) if args.cache is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.images:
        raise ValueError('Please provide at least one image path via --images')

    for image_path in args.images:
        image_path = Path(image_path)
        pil = Image.open(image_path).convert('RGB')
        tensor = transform(pil).unsqueeze(0).to(device)
        explain = model.explain(tensor)
        pred = int(explain['pred'][0].item())
        logits = explain['logits'][0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        visible_mask = explain['visible_masks'][0].detach().cpu().numpy().astype(bool)
        per_dir_maps = explain['per_direction_maps'][0].detach().cpu()
        aggregate = explain['aggregate_heatmaps'][0].detach().cpu()
        aggregate = torch.nn.functional.interpolate(
            aggregate[None, None],
            size=(config.model.img_size, config.model.img_size),
            mode='bilinear',
            align_corners=False,
        )[0, 0]
        aggregate = aggregate - aggregate.min()
        aggregate = aggregate / (aggregate.max() + 1e-6)
        aggregate_np = aggregate.numpy()
        img_np = _denormalize(tensor[0])
        overlay_np = _overlay(img_np, aggregate_np)

        visible_indices = np.where(visible_mask)[0].tolist()
        n_rows = 2 + len(visible_indices)
        n_cols = 1 + args.top_m
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes[:, None]

        for ax in axes.ravel():
            ax.axis('off')

        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Input\nPred: {class_names[pred]}')
        axes[0, 0].axis('on')
        axes[0, 1].imshow(overlay_np)
        axes[0, 1].set_title('Aggregate heatmap')
        axes[0, 1].axis('on')
        topk = np.argsort(-probs)[: min(5, len(probs))]
        txt = '\n'.join([f'{class_names[k]}: {probs[k]:.3f}' for k in topk])
        axes[0, min(2, n_cols - 1)].text(0.0, 0.95, txt, fontsize=11, va='top')
        axes[0, min(2, n_cols - 1)].set_title('Top logits')
        axes[0, min(2, n_cols - 1)].axis('off')

        axes[1, 0].imshow(aggregate_np, cmap='jet')
        axes[1, 0].set_title('Aggregate map')
        axes[1, 0].axis('on')

        active_score_text = []
        dir_scores = explain['per_direction_scores'][0].detach().cpu().numpy()
        for k in visible_indices:
            active_score_text.append(f'k={k}: {dir_scores[k]:.3f}')
        axes[1, 1].text(0.0, 0.95, '\n'.join(active_score_text[:12]), fontsize=10, va='top')
        axes[1, 1].set_title('Visible directions')
        axes[1, 1].axis('off')

        for row_idx, k in enumerate(visible_indices, start=2):
            dir_map = per_dir_maps[k]
            dir_map = torch.nn.functional.interpolate(
                dir_map[None, None],
                size=(config.model.img_size, config.model.img_size),
                mode='bilinear',
                align_corners=False,
            )[0, 0]
            dir_map = dir_map - dir_map.min()
            dir_map = dir_map / (dir_map.max() + 1e-6)
            dir_np = dir_map.numpy()
            axes[row_idx, 0].imshow(_overlay(img_np, dir_np))
            axes[row_idx, 0].set_title(f'Direction {k}')
            axes[row_idx, 0].axis('on')

            if cache is not None:
                exemplars = cache['cache'].get(pred, cache['cache'].get(str(pred), {}))
                exemplars = exemplars.get(k, exemplars.get(str(k), []))
                for col_idx, entry in enumerate(exemplars[:args.top_m], start=1):
                    patch = _load_exemplar_patch(entry)
                    axes[row_idx, col_idx].imshow(patch)
                    axes[row_idx, col_idx].set_title(f"Exemplar {col_idx}\nscore={entry['score']:.3f}")
                    axes[row_idx, col_idx].axis('on')

        fig.tight_layout()
        save_path = output_dir / f'{image_path.stem}_explanation.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()
