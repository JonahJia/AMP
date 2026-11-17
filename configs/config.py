from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml


def _convert_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_paths(v) for v in obj]
    return obj


@dataclass
class ModelConfig:
    base_architecture: Literal[
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'toy_cnn'
    ] = 'resnet50'
    pretrained: bool = True
    img_size: int = 448
    num_classes: int = 200
    prototype_dim: int = 128
    subspace_dim: int = 10
    active_threshold: float = 1e-6
    explain_threshold: float = 1e-4
    add_on_layers_type: Literal['bottleneck', 'regular'] = 'regular'
    eps: float = 1e-6


@dataclass
class DataConfig:
    data_root: Path = Path('./datasets/cub200')
    train_dir: str = 'train'
    test_dir: str = 'test'
    train_push_dir: str = 'train'
    train_batch_size: int = 32
    test_batch_size: int = 32
    train_push_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    crop_to_bbox: bool = False
    bbox_file: Optional[Path] = None
    mask_root: Optional[Path] = None
    mask_extension: str = '.png'
    part_annotations_file: Optional[Path] = None
    return_paths: bool = True

    @property
    def train_path(self) -> Path:
        return self.data_root / self.train_dir

    @property
    def test_path(self) -> Path:
        return self.data_root / self.test_dir

    @property
    def train_push_path(self) -> Path:
        return self.data_root / self.train_push_dir


@dataclass
class OptimizerConfig:
    euclidean_optimizer: Literal['sgd', 'adamw'] = 'sgd'
    backbone_lr: float = 1e-3
    add_on_lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    basis_lr: float = 5e-3
    capacity_lr: float = 5e-3
    scheduler: Literal['cosine', 'step', 'none'] = 'cosine'
    min_lr: float = 1e-5
    step_size: int = 30
    gamma: float = 0.1
    freeze_backbone_epochs: int = 0
    max_grad_norm: Optional[float] = None


@dataclass
class LossConfig:
    gamma_sem: float = 1e-2
    gamma_overlap: float = 1e-2
    lambda_sparse: float = 1e-4


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    eval_every: int = 1
    save_every: int = 5
    log_every: int = 20
    save_threshold: float = 0.0
    deterministic: bool = True
    build_cache_on_fit_end: bool = True


@dataclass
class RetrievalConfig:
    top_m: int = 5
    cache_filename: str = 'exemplar_cache.pt'
    percentile: int = 95


@dataclass
class EvalConfig:
    dauc_steps: int = 20
    perturb_sigma: float = 0.2
    consistency_threshold: float = 0.8
    grid_size: int = 3
    correct_only: bool = True
    deletion_batch_size: int = 8


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = 'default'
    seed: int = 42
    device: str = 'cuda'

    @property
    def output_dir(self) -> Path:
        return Path('./runs') / self.model.base_architecture / self.experiment_name

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        data_cfg = dict(config_dict.get('data', {}))
        for key in ['data_root', 'bbox_file', 'mask_root', 'part_annotations_file']:
            if key in data_cfg and data_cfg[key] is not None:
                data_cfg[key] = Path(data_cfg[key])
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**data_cfg),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            evaluation=EvalConfig(**config_dict.get('evaluation', {})),
            **{k: v for k, v in config_dict.items() if k not in {
                'model', 'data', 'optimizer', 'loss', 'training', 'retrieval', 'evaluation'
            }}
        )

    def to_yaml(self, yaml_path: str) -> None:
        config_dict: Dict[str, Any] = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'optimizer': asdict(self.optimizer),
            'loss': asdict(self.loss),
            'training': asdict(self.training),
            'retrieval': asdict(self.retrieval),
            'evaluation': asdict(self.evaluation),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(_convert_paths(config_dict), f, sort_keys=False)


def get_default_config() -> Config:
    return Config()
