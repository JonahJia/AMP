from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from configs.config import Config
from src.analysis import compute_collapse_diagnostics
from src.data import create_dataloaders
from src.metrics import evaluate_interpretability
from src.model import build_model
from src.trainer import Trainer
from utils.helpers import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate AMP checkpoints')
    parser.add_argument('--config', type=str, required=True, help='AMP config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to AMP checkpoint')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output', type=str, default=None, help='Path to JSON metrics output')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'push'], help='Dataset split for evaluation')
    parser.add_argument('--skip-interpretability', action='store_true', help='Skip interpretability metrics')
    parser.add_argument('--skip-collapse', action='store_true', help='Skip collapse diagnostics')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    config = Config.from_yaml(args.config)
    device = config.device if torch.cuda.is_available() and config.device.startswith('cuda') else 'cpu'
    config.device = device

    output_dir = Path(args.output).parent if args.output is not None else Path(args.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('evaluate', output_dir / 'evaluate.log')

    train_loader, test_loader, push_loader, data_module = create_dataloaders(config)
    eval_loader = test_loader if args.split == 'test' else push_loader

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
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    trainer = Trainer(model, device=device, logger=logger)

    metrics = trainer.evaluate_accuracy(eval_loader, desc=f'{args.split.capitalize()} accuracy')
    logger.info(f'Accuracy metrics: {metrics}')

    if not args.skip_interpretability:
        interp = evaluate_interpretability(trainer.model, eval_loader, config.evaluation, device=device)
        logger.info(f'Interpretability metrics: {interp}')
        metrics.update({f'interp_{k}': v for k, v in interp.items()})

    if not args.skip_collapse:
        collapse = compute_collapse_diagnostics(trainer.model, eval_loader, device=device)
        logger.info(f'Collapse diagnostics: {collapse}')
        metrics.update({f'collapse_{k}': v for k, v in collapse.items()})

    out_path = Path(args.output) if args.output is not None else output_dir / 'evaluation_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Saved metrics to: {out_path}')

if __name__ == '__main__':
    main()
