from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .riemannian import prox_nonnegative_l1_step, stiefel_batch_sgd_step


@dataclass
class TrainEpochMetrics:
    total_objective: float
    backward_loss: float
    cross_entropy: float
    sem: float
    overlap: float
    sparse_penalty: float
    accuracy: float


class Trainer:
    def __init__(self, model, device: str = 'cuda', logger: Optional[logging.Logger] = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = trainable

    def create_euclidean_optimizer(self, config):
        param_groups = [
            {
                'params': list(self.model.features.parameters()),
                'lr': config.optimizer.backbone_lr,
                'weight_decay': config.optimizer.weight_decay,
            },
            {
                'params': list(self.model.add_on_layers.parameters()),
                'lr': config.optimizer.add_on_lr,
                'weight_decay': config.optimizer.weight_decay,
            },
        ]
        if config.optimizer.euclidean_optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=config.optimizer.momentum,
            )
        elif config.optimizer.euclidean_optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer.euclidean_optimizer}")

        if config.optimizer.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.optimizer.min_lr,
            )
        elif config.optimizer.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.optimizer.step_size,
                gamma=config.optimizer.gamma,
            )
        else:
            scheduler = None
        return optimizer, scheduler

    def compute_losses(self, outputs, labels, config) -> Dict[str, torch.Tensor]:
        cross_entropy = F.cross_entropy(outputs.logits, labels)
        reg = self.model.compute_spatial_regularizers(outputs.basis_maps, labels)
        sem = reg['sem']
        overlap = reg['overlap']
        sparse_penalty = outputs.capacities.sum()
        backward_loss = cross_entropy + config.loss.gamma_sem * sem + config.loss.gamma_overlap * overlap
        total_objective = backward_loss + config.loss.lambda_sparse * sparse_penalty
        return {
            'cross_entropy': cross_entropy,
            'sem': sem,
            'overlap': overlap,
            'sparse_penalty': sparse_penalty,
            'backward_loss': backward_loss,
            'total_objective': total_objective,
        }

    def train_epoch(self, dataloader, optimizer, config, epoch: int) -> TrainEpochMetrics:
        self.model.train()
        freeze_backbone = epoch < config.optimizer.freeze_backbone_epochs
        self.set_backbone_trainable(not freeze_backbone)

        meters = {
            'total_objective': 0.0,
            'backward_loss': 0.0,
            'cross_entropy': 0.0,
            'sem': 0.0,
            'overlap': 0.0,
            'sparse_penalty': 0.0,
        }
        num_correct = 0
        num_total = 0

        pbar = tqdm(dataloader, desc=f'Train {epoch:03d}', leave=False)
        for step, (images, labels, metas) in enumerate(pbar, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(images)
            losses = self.compute_losses(outputs, labels, config)
            losses['backward_loss'].backward()

            if config.optimizer.max_grad_norm is not None:
                euclid_params = [
                    p for p in list(self.model.features.parameters()) + list(self.model.add_on_layers.parameters())
                    if p.requires_grad and p.grad is not None
                ]
                if euclid_params:
                    torch.nn.utils.clip_grad_norm_(euclid_params, config.optimizer.max_grad_norm)

            optimizer.step()

            with torch.no_grad():
                if self.model.class_bases.grad is not None:
                    self.model.class_bases.copy_(
                        stiefel_batch_sgd_step(
                            self.model.class_bases.detach(),
                            self.model.class_bases.grad.detach(),
                            lr=config.optimizer.basis_lr,
                        )
                    )
                if self.model.capacities.grad is not None:
                    self.model.capacities.copy_(
                        prox_nonnegative_l1_step(
                            self.model.capacities.detach(),
                            self.model.capacities.grad.detach(),
                            lr=config.optimizer.capacity_lr,
                            lam=config.loss.lambda_sparse,
                        )
                    )

            preds = outputs.logits.argmax(dim=1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)

            for key in meters:
                meters[key] += float(losses[key].detach().item())

            if step % max(config.training.log_every, 1) == 0 or step == len(dataloader):
                pbar.set_postfix({
                    'loss': losses['total_objective'].item(),
                    'acc': f'{100.0 * num_correct / max(num_total,1):.2f}',
                    'active': f'{self.model.get_active_mask().sum(dim=1).float().mean().item():.2f}',
                })

            self.model.zero_grad(set_to_none=True)

        n = max(len(dataloader), 1)
        return TrainEpochMetrics(
            total_objective=meters['total_objective'] / n,
            backward_loss=meters['backward_loss'] / n,
            cross_entropy=meters['cross_entropy'] / n,
            sem=meters['sem'] / n,
            overlap=meters['overlap'] / n,
            sparse_penalty=meters['sparse_penalty'] / n,
            accuracy=100.0 * num_correct / max(num_total, 1),
        )

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader, desc: str = 'Eval') -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for images, labels, metas in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = F.cross_entropy(outputs.logits, labels)
            preds = outputs.logits.argmax(dim=1)
            total_loss += float(loss.item())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        return {
            'loss': total_loss / max(len(dataloader), 1),
            'accuracy': 100.0 * total_correct / max(total_samples, 1),
            'avg_active_rank': float(self.model.get_active_mask().sum(dim=1).float().mean().item()),
        }

    def save_checkpoint(self, path: str | Path, epoch: int, optimizer, scheduler, metrics: Dict[str, float], config) -> None:
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'metrics': metrics,
            'config': {
                'experiment_name': config.experiment_name,
                'model': config.model.__dict__,
                'data': {k: str(v) if isinstance(v, Path) else v for k, v in config.data.__dict__.items()},
                'optimizer': config.optimizer.__dict__,
                'loss': config.loss.__dict__,
                'training': config.training.__dict__,
                'retrieval': config.retrieval.__dict__,
                'evaluation': config.evaluation.__dict__,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def fit(self, train_loader, test_loader, config) -> Dict[str, float]:
        optimizer, scheduler = self.create_euclidean_optimizer(config)
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        best_acc = -1.0
        best_metrics: Dict[str, float] = {}
        history = []

        for epoch in range(1, config.training.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, optimizer, config, epoch)
            log_line = (
                f'Epoch {epoch:03d} | '
                f'train_acc={train_metrics.accuracy:.2f} | '
                f'train_obj={train_metrics.total_objective:.4f} | '
                f'ce={train_metrics.cross_entropy:.4f} | '
                f'sem={train_metrics.sem:.4f} | '
                f'ov={train_metrics.overlap:.4f} | '
                f'sparse={train_metrics.sparse_penalty:.4f}'
            )
            self.logger.info(log_line)

            if scheduler is not None:
                scheduler.step()

            epoch_record: Dict[str, float] = {
                'epoch': epoch,
                'train_accuracy': train_metrics.accuracy,
                'train_total_objective': train_metrics.total_objective,
                'train_cross_entropy': train_metrics.cross_entropy,
                'train_sem': train_metrics.sem,
                'train_overlap': train_metrics.overlap,
                'train_sparse_penalty': train_metrics.sparse_penalty,
            }

            if epoch % config.training.eval_every == 0:
                eval_metrics = self.evaluate_accuracy(test_loader, desc=f'Test {epoch:03d}')
                self.logger.info(
                    f"Epoch {epoch:03d} | test_acc={eval_metrics['accuracy']:.2f} | "
                    f"test_loss={eval_metrics['loss']:.4f} | avg_active_rank={eval_metrics['avg_active_rank']:.2f}"
                )
                epoch_record.update({f'test_{k}': v for k, v in eval_metrics.items()})
                if eval_metrics['accuracy'] > best_acc and eval_metrics['accuracy'] >= 100.0 * config.training.save_threshold:
                    best_acc = eval_metrics['accuracy']
                    best_metrics = eval_metrics
                    self.save_checkpoint(output_dir / 'best.pt', epoch, optimizer, scheduler, eval_metrics, config)

            if epoch % config.training.save_every == 0:
                self.save_checkpoint(output_dir / f'epoch_{epoch:03d}.pt', epoch, optimizer, scheduler, epoch_record, config)

            history.append(epoch_record)
            with open(output_dir / 'history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        if best_acc < 0:
            # If no evaluation ran, save the final checkpoint.
            best_metrics = {'accuracy': train_metrics.accuracy}
            self.save_checkpoint(output_dir / 'best.pt', config.training.num_epochs, optimizer, scheduler, best_metrics, config)

        return best_metrics
