#!/usr/bin/env python3
"""Build real tiny ImageFolder train/val loaders through the delivery path."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from PIL import Image

from opentome.utils.dataset_loader import build_dataset


def write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), color=color).save(path)


def main():
    with TemporaryDirectory(prefix="mergenet-imagefolder-") as tmp:
        root = Path(tmp)
        for split in ("train", "val"):
            write_image(root / split / "class_a" / "a.jpg", (255, 0, 0))
            write_image(root / split / "class_b" / "b.jpg", (0, 255, 0))

        args = SimpleNamespace(
            dataset="imagefolder",
            data_dir=str(root),
            train_split="train",
            val_split="val",
            class_map="",
            dataset_download=False,
            batch_size=1,
            validation_batch_size=1,
            epoch_repeats=0.0,
            debug_subset=0,
            no_aug=False,
            train_interpolation="bicubic",
            prefetcher=False,
            reprob=0.0,
            remode="pixel",
            recount=1,
            resplit=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333],
            hflip=0.0,
            vflip=0.0,
            color_jitter=0.0,
            aa=None,
            aug_repeats=0,
            workers=0,
            distributed=False,
            pin_mem=False,
            use_multi_epochs_loader=False,
            worker_seeding="all",
            num_classes=2,
        )
        data_config = {
            "input_size": (3, 32, 32),
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "crop_pct": 0.9,
        }
        train_loader, val_loader = build_dataset(
            args, data_config, collate_fn=None, num_aug_splits=0
        )
        assert len(train_loader.dataset) == 2
        assert len(val_loader.dataset) == 2
        train_images, train_targets = next(iter(train_loader))
        val_images, val_targets = next(iter(val_loader))
        assert train_images.shape == (1, 3, 32, 32)
        assert val_images.shape == (1, 3, 32, 32)
        assert train_targets.ndim == val_targets.ndim == 1

    print("IMAGEFOLDER_TRAIN_VAL_TEST_PASS")


if __name__ == "__main__":
    main()
