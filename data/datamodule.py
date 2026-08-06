"""Lightning DataModule wrapping clean CIFAR-10 (train/val) and CIFAR-10-C (test)."""

from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .cifar10c import CIFAR10C, CORRUPTIONS

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, root: str = "data", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        self.train_set = None
        self.val_set = None

    def setup(self, stage: str | None = None) -> None:
        cifar_root = f"{self.root}/cifar10"
        if stage in ("fit", None):
            self.train_set = CIFAR10(cifar_root, train=True, transform=self.train_transform, download=False)
        if stage in ("fit", "validate", None):
            self.val_set = CIFAR10(cifar_root, train=False, transform=self.eval_transform, download=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def clean_test_dataloader(self) -> DataLoader:
        """CIFAR-10 test set, treated as the "severity 0" / in-distribution case."""
        return self.val_dataloader()

    def corrupted_dataloader(self, corruption: str, severity: int) -> DataLoader:
        dataset = CIFAR10C(
            f"{self.root}/cifar10-c", corruption, severity, transform=self.eval_transform
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    @staticmethod
    def corruptions() -> list[str]:
        return list(CORRUPTIONS)
