from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from configs.config import Config, get_default_config
from src.analysis import build_exemplar_cache, save_exemplar_cache
from src.data import create_dataloaders
from src.model import build_model
from src.trainer import Trainer
from utils.helpers import count_parameters, set_seed, setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Adaptive Manifold Prototypes (AMP)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to AMP YAML config')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--no-cache', action='store_true', help='Do not build exemplar cache at the end')
    return parser.parse_args()

def load_config(path: str | None) -> Config:
    if path is None:
        return get_default_config()
    return Config.from_yaml(path)

def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    device = config.device if torch.cuda.is_available() and config.device.startswith('cuda') else 'cpu'
    config.device = device

    set_seed(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('train', output_dir / 'train.log')
    config.to_yaml(output_dir / 'config.yaml')

    logger.info('=' * 80)
    logger.info('Starting AMP training')
    logger.info('=' * 80)
    logger.info(f'Config file: {args.config}')
    logger.info(f'Experiment: {config.experiment_name}')
    logger.info(f'Device: {config.device}')
    logger.info(f'Seed: {config.seed}')

    train_loader, test_loader, push_loader, data_module = create_dataloaders(config)
    logger.info(f'Train set size: {len(train_loader.dataset)}')
    logger.info(f'Test set size: {len(test_loader.dataset)}')
    logger.info(f'Push set size: {len(push_loader.dataset)}')

    model = build_model(
        base_architecture=config.model.base_architecture,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        prototype_dim=config.model.prototype_dim,
        subspace_dim=config.model.subspace_dim,
        img_size=config.model.img_size,
        active_threshold=config.model.active_threshold,
        explain_threshold=config.model.explain_threshold,
        add_on_layers_type=config.model.add_on_layers_type,
        eps=config.model.eps,
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    trainer = Trainer(model=model, device=config.device, logger=logger)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=config.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f'Resumed model weights from: {args.resume}')

    best_metrics = trainer.fit(train_loader, test_loader, config)
    logger.info(f'Best metrics: {best_metrics}')

    if not args.no_cache and config.training.build_cache_on_fit_end:
        logger.info('Building exemplar cache from the push set...')
        best_ckpt_path = output_dir / 'best.pt'
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=config.device)
            trainer.model.load_state_dict(ckpt['model_state_dict'])
        cache = build_exemplar_cache(
            trainer.model,
            push_loader,
            device=config.device,
            top_m=config.retrieval.top_m,
        )
        cache_path = output_dir / config.retrieval.cache_filename
        save_exemplar_cache(cache, cache_path)
        logger.info(f'Saved exemplar cache to: {cache_path}')

    logger.info('Training completed successfully.')


if __name__ == '__main__':
    main()
