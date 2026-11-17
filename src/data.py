from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json

from PIL import Image
import torch
from torch.utils.data import DataLoader
import utils.torchvision_compat  # noqa: F401
from torchvision import datasets, transforms

BBox = Tuple[float, float, float, float]

def _to_path(x: Optional[str | Path]) -> Optional[Path]:
    return None if x is None else Path(x)


class AnnotationIndex:
    """Lightweight lookup for optional bbox / mask / part annotations."""

    def __init__(
        self,
        bbox_file: Optional[str | Path] = None,
        mask_root: Optional[str | Path] = None,
        mask_extension: str = '.png',
        part_annotations_file: Optional[str | Path] = None,
    ) -> None:
        self.bbox_file = _to_path(bbox_file)
        self.mask_root = _to_path(mask_root)
        self.mask_extension = mask_extension
        self.part_annotations_file = _to_path(part_annotations_file)
        self.bboxes: Dict[str, BBox] = {}
        self.parts: Dict[str, List[Dict[str, Any]]] = {}
        if self.bbox_file is not None:
            self.bboxes = self._load_bboxes(self.bbox_file)
        if self.part_annotations_file is not None:
            self.parts = self._load_parts(self.part_annotations_file)

    def _load_bboxes(self, path: Path) -> Dict[str, BBox]:
        if path.suffix.lower() == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            return {k: tuple(map(float, v)) for k, v in data.items()}
        rows: Dict[str, BBox] = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get('path') or row.get('image') or row.get('filename')
                if key is None:
                    raise ValueError('BBox CSV must contain a path/image/filename column')
                x1 = float(row['x1'])
                y1 = float(row['y1'])
                x2 = float(row['x2'])
                y2 = float(row['y2'])
                rows[key] = (x1, y1, x2, y2)
        return rows

    def _load_parts(self, path: Path) -> Dict[str, List[Dict[str, Any]]]:
        if path.suffix.lower() == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            out: Dict[str, List[Dict[str, Any]]] = {}
            for k, parts in data.items():
                norm_parts = []
                for p in parts:
                    if isinstance(p, dict):
                        norm_parts.append({
                            'name': p.get('name', p.get('label', 'part')),
                            'x': float(p['x']),
                            'y': float(p['y']),
                            'visible': int(p.get('visible', 1)),
                        })
                    else:
                        name, x, y, *rest = p
                        norm_parts.append({
                            'name': str(name),
                            'x': float(x),
                            'y': float(y),
                            'visible': int(rest[0]) if rest else 1,
                        })
                out[k] = norm_parts
            return out

        out: Dict[str, List[Dict[str, Any]]] = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get('path') or row.get('image') or row.get('filename')
                if key is None:
                    raise ValueError('Part CSV must contain a path/image/filename column')
                out.setdefault(key, []).append({
                    'name': row.get('name', row.get('label', 'part')),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'visible': int(row.get('visible', 1)),
                })
        return out

    def _candidate_keys(self, image_path: Path, split_root: Path, data_root: Optional[Path] = None) -> List[str]:
        keys = []
        try:
            keys.append(image_path.relative_to(split_root).as_posix())
        except ValueError:
            pass
        if data_root is not None:
            try:
                keys.append(image_path.relative_to(data_root).as_posix())
            except ValueError:
                pass
        keys.append(image_path.name)
        return list(dict.fromkeys(keys))

    def get_bbox(self, image_path: Path, split_root: Path, data_root: Optional[Path] = None) -> Optional[BBox]:
        for key in self._candidate_keys(image_path, split_root, data_root):
            if key in self.bboxes:
                return self.bboxes[key]
        return None

    def get_parts(self, image_path: Path, split_root: Path, data_root: Optional[Path] = None) -> Optional[List[Dict[str, Any]]]:
        for key in self._candidate_keys(image_path, split_root, data_root):
            if key in self.parts:
                return self.parts[key]
        return None

    def get_mask_path(self, image_path: Path, split_root: Path, data_root: Optional[Path] = None) -> Optional[Path]:
        if self.mask_root is None:
            return None
        for key in self._candidate_keys(image_path, split_root, data_root):
            rel = Path(key)
            if rel.suffix:
                rel = rel.with_suffix(self.mask_extension)
            else:
                rel = rel.parent / f'{rel.name}{self.mask_extension}'
            candidate = self.mask_root / rel
            if candidate.exists():
                return candidate
        return None


class ImageFolderWithMetadata(datasets.ImageFolder):
    """ImageFolder dataset with optional bbox cropping and metadata."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[transforms.Compose] = None,
        annotation_index: Optional[AnnotationIndex] = None,
        data_root: Optional[str | Path] = None,
        crop_to_bbox: bool = False,
        return_metadata: bool = True,
    ) -> None:
        super().__init__(str(root), transform=transform)
        self.root_path = Path(root)
        self.data_root = _to_path(data_root)
        self.annotation_index = annotation_index or AnnotationIndex()
        self.crop_to_bbox = crop_to_bbox
        self.return_metadata = return_metadata

    def __getitem__(self, index: int):
        path_str, target = self.samples[index]
        path = Path(path_str)
        image = self.loader(str(path)).convert('RGB')
        orig_w, orig_h = image.size

        bbox = self.annotation_index.get_bbox(path, self.root_path, self.data_root)
        crop_bbox = None
        if bbox is not None and self.crop_to_bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(round(x1)), orig_w - 1))
            y1 = max(0, min(int(round(y1)), orig_h - 1))
            x2 = max(x1 + 1, min(int(round(x2)), orig_w))
            y2 = max(y1 + 1, min(int(round(y2)), orig_h))
            crop_bbox = (x1, y1, x2, y2)
            image = image.crop(crop_bbox)

        if self.transform is not None:
            image = self.transform(image)

        if not self.return_metadata:
            return image, target, {}

        try:
            rel_path = path.relative_to(self.root_path).as_posix()
        except ValueError:
            rel_path = path.name

        meta = {
            'path': str(path),
            'rel_path': rel_path,
            'label': int(target),
            'original_size': (orig_w, orig_h),
            'bbox': tuple(map(float, bbox)) if bbox is not None else None,
            'crop_bbox': crop_bbox,
            'mask_path': str(self.annotation_index.get_mask_path(path, self.root_path, self.data_root))
            if self.annotation_index.get_mask_path(path, self.root_path, self.data_root) is not None else None,
            'parts': self.annotation_index.get_parts(path, self.root_path, self.data_root),
            'crop_to_bbox': self.crop_to_bbox,
        }
        return image, target, meta


def collate_with_metadata(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return images, labels, metadata


@dataclass
class DataModule:
    data_root: Path
    train_dir: str
    test_dir: str
    train_push_dir: str
    img_size: int = 448
    train_batch_size: int = 32
    test_batch_size: int = 32
    train_push_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    crop_to_bbox: bool = False
    bbox_file: Optional[str | Path] = None
    mask_root: Optional[str | Path] = None
    mask_extension: str = '.png'
    part_annotations_file: Optional[str | Path] = None

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.annotation_index = AnnotationIndex(
            bbox_file=self.bbox_file,
            mask_root=self.mask_root,
            mask_extension=self.mask_extension,
            part_annotations_file=self.part_annotations_file,
        )

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        self.push_transform = self.eval_transform

    @property
    def train_path(self) -> Path:
        return self.data_root / self.train_dir

    @property
    def test_path(self) -> Path:
        return self.data_root / self.test_dir

    @property
    def train_push_path(self) -> Path:
        return self.data_root / self.train_push_dir

    def get_train_loader(self) -> DataLoader:
        dataset = ImageFolderWithMetadata(
            self.train_path,
            transform=self.train_transform,
            annotation_index=self.annotation_index,
            data_root=self.data_root,
            crop_to_bbox=self.crop_to_bbox,
            return_metadata=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_with_metadata,
        )

    def get_test_loader(self) -> DataLoader:
        dataset = ImageFolderWithMetadata(
            self.test_path,
            transform=self.eval_transform,
            annotation_index=self.annotation_index,
            data_root=self.data_root,
            crop_to_bbox=self.crop_to_bbox,
            return_metadata=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_with_metadata,
        )

    def get_push_loader(self) -> DataLoader:
        dataset = ImageFolderWithMetadata(
            self.train_push_path,
            transform=self.push_transform,
            annotation_index=self.annotation_index,
            data_root=self.data_root,
            crop_to_bbox=self.crop_to_bbox,
            return_metadata=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_push_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_with_metadata,
        )

    @property
    def num_classes(self) -> int:
        dataset = datasets.ImageFolder(self.train_path)
        return len(dataset.classes)

    def get_class_names(self) -> List[str]:
        dataset = datasets.ImageFolder(self.train_path)
        return dataset.classes


def create_dataloaders(config):
    data_module = DataModule(
        data_root=config.data.data_root,
        train_dir=config.data.train_dir,
        test_dir=config.data.test_dir,
        train_push_dir=config.data.train_push_dir,
        img_size=config.model.img_size,
        train_batch_size=config.data.train_batch_size,
        test_batch_size=config.data.test_batch_size,
        train_push_batch_size=config.data.train_push_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        crop_to_bbox=config.data.crop_to_bbox,
        bbox_file=config.data.bbox_file,
        mask_root=config.data.mask_root,
        mask_extension=config.data.mask_extension,
        part_annotations_file=config.data.part_annotations_file,
    )
    return data_module.get_train_loader(), data_module.get_test_loader(), data_module.get_push_loader(), data_module
