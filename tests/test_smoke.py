from __future__ import annotations
import sys
from pathlib import Path
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import get_default_config
from src.model import build_model
from src.trainer import Trainer

class DummyLoader:
    def __init__(self, n_batches=1, batch_size=1, num_classes=4, img_size=64):
        self.batches = []
        for _ in range(n_batches):
            images = torch.randn(batch_size, 3, img_size, img_size)
            labels = torch.randint(0, num_classes, (batch_size,))
            metas = [{'path': 'dummy.jpg', 'rel_path': 'dummy.jpg', 'label': int(labels[i]), 'original_size': (img_size, img_size), 'bbox': None, 'crop_bbox': None, 'mask_path': None, 'parts': None, 'crop_to_bbox': False} for i in range(batch_size)]
            self.batches.append((images, labels, metas))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def main() -> None:
    config = get_default_config()
    config.model.base_architecture = 'toy_cnn'
    config.model.pretrained = False
    config.model.num_classes = 4
    config.model.prototype_dim = 16
    config.model.subspace_dim = 3
    config.model.img_size = 32
    config.data.train_batch_size = 2
    config.data.test_batch_size = 2
    config.training.num_epochs = 1
    config.training.eval_every = 1
    config.training.save_every = 1
    config.training.log_every = 1
    config.training.build_cache_on_fit_end = False
    config.optimizer.backbone_lr = 1e-4
    config.optimizer.add_on_lr = 1e-3
    config.optimizer.basis_lr = 1e-3
    config.optimizer.capacity_lr = 1e-3
    config.device = 'cpu'
    config.experiment_name = 'smoke_test'

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
    trainer = Trainer(model, device='cpu')
    train_loader = DummyLoader(n_batches=1, batch_size=1, num_classes=config.model.num_classes, img_size=config.model.img_size)
    test_loader = DummyLoader(n_batches=1, batch_size=1, num_classes=config.model.num_classes, img_size=config.model.img_size)
    best = trainer.fit(train_loader, test_loader, config)
    assert 'accuracy' in best
    sample = torch.randn(2, 3, config.model.img_size, config.model.img_size)
    out = trainer.model(sample)
    assert out.logits.shape == (2, config.model.num_classes)
    assert out.basis_maps.shape[:3] == (2, config.model.num_classes, config.model.subspace_dim)
    print('AMP smoke test passed.')

if __name__ == '__main__':
    main()
