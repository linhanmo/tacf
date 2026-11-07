from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SPLIT_NAMES, TACFDataset


__all__ = ["TACFDataModule", "build_dataloaders"]


@dataclass
class DataModuleStats:
    n_features: int
    stamp_dim: int
    feature_names: List[str]
    target_name: str
    n_train: int
    n_val: int
    n_test: int


class TACFDataModule:
    """Lightweight data-module wrapper holding train/val/test DataLoaders.

    Parameters
    ----------
    root : str
        Path to the ``preprocess`` folder containing per-dataset subfolders.
    dataset : str
        Dataset name (e.g. ``ETTh1``, ``electricity``, ``traffic``).
    seq_len : int
        Look-back (input) window length.
    pred_len : int
        Forecast (output) horizon length.
    label_len : int
        Label warm-start length (kept for compatibility, not used in MOA).
    batch_size : int
        Per-device batch size.
    num_workers : int
        Dataloader workers (0 = inline loading).
    pin_memory : bool
        Pin memory when GPU is available.
    drop_last : bool
        Drop the last incomplete train batch.
    shuffle : bool
        Shuffle training split (val/test are never shuffled).
    use_time_features : bool
        Whether to load continuous time-stamp features.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        seq_len: int,
        pred_len: int,
        label_len: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        use_time_features: bool = True,
    ) -> None:
        self.root = root
        self.dataset_name = dataset
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.label_len = int(label_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.use_time_features = bool(use_time_features)

        self._datasets: Dict[str, TACFDataset] = {}
        for split in SPLIT_NAMES:
            self._datasets[split] = TACFDataset(
                root=root,
                dataset=dataset,
                split=split,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                label_len=self.label_len,
                use_time_features=self.use_time_features,
            )

        # ---- Emergency train/val/test reslicing fix for small datasets + long
        #      horizons (e.g. exchange_rate s512 → p720): the precomputed split
        #      files can end up with 0 windows in val/test, which then causes
        #      trainer._evaluate to hit torch.cat([]) ValueError. Reslice the
        #      original train split's WINDOWS into a 65/15/20 temporal holdout.
        try:
            n_train_wins = len(self._datasets["train"])
        except Exception:
            n_train_wins = 0
        try:
            n_val_wins = len(self._datasets["val"])
        except Exception:
            n_val_wins = 0
        try:
            n_test_wins = len(self._datasets["test"])
        except Exception:
            n_test_wins = 0
        MIN_WINDOWS = 4
        if n_train_wins >= MIN_WINDOWS * 3 and (
            n_val_wins < MIN_WINDOWS or n_test_wins < MIN_WINDOWS
        ):
            from .dataset import TACFDataset as _TACFDS

            base_train = self._datasets["train"]
            total_rows = base_train.data.shape[0]
            # Window split order: |-- train (65%) --|-- val (15%) --|-- test (20%) --|.
            train_w_end = max(MIN_WINDOWS, int(n_train_wins * 0.65))
            val_w_end   = max(train_w_end + MIN_WINDOWS, int(n_train_wins * 0.80))
            test_w_end  = n_train_wins
            # Translate each window-range into row-range: window i covers rows
            # [i : i + seq_len + pred_len).
            SPAN = self.seq_len + self.pred_len

            def _rewrap(w_start: int, w_end: int, tag_split: str) -> TACFDataset:
                ws = int(max(0, min(n_train_wins, w_start)))
                we = int(max(ws, min(n_train_wins, w_end)))
                rs = ws
                re = min(total_rows, we + SPAN)
                # Construct a fully independent TACFDataset.
                ds = _TACFDS.__new__(_TACFDS)
                ds.root = base_train.root
                ds.dataset = base_train.dataset
                ds.split = tag_split
                ds.seq_len = base_train.seq_len
                ds.pred_len = base_train.pred_len
                ds.label_len = base_train.label_len
                ds.use_time_features = base_train.use_time_features
                ds.folder = base_train.folder
                ds.data = base_train.data[rs:re].clone()
                if base_train.stamp is not None and 0 <= rs <= re <= base_train.stamp.shape[0]:
                    ds.stamp = base_train.stamp[rs:re].clone()
                else:
                    ds.stamp = base_train.stamp[rs:re].clone() if base_train.stamp is not None else torch.zeros(re - rs, 0)
                ds.meta = dict(base_train.meta)
                ds.scaler = base_train.scaler   # keep train-fitted StandardScaler
                ds.n_samples = max(0, ds.data.shape[0] - ds.seq_len - ds.pred_len + 1)
                return ds

            self._datasets["train"] = _rewrap(0,            train_w_end, "train")
            self._datasets["val"]   = _rewrap(train_w_end,  val_w_end,   "val")
            self._datasets["test"]  = _rewrap(val_w_end,    test_w_end,  "test")

        meta = self.train_dataset.meta
        self.n_features: int = self.train_dataset.n_features
        self.stamp_dim: int = self.train_dataset.stamp_dim
        self.feature_names: List[str] = meta.get(
            "feature_names", [f"f{i}" for i in range(self.n_features)]
        )
        self.target_name: str = meta.get("target", "OT")
        self.scaler = self.train_dataset.scaler

    @property
    def stats(self) -> DataModuleStats:
        return DataModuleStats(
            n_features=self.n_features,
            stamp_dim=self.stamp_dim,
            feature_names=list(self.feature_names),
            target_name=self.target_name,
            n_train=len(self.train_dataset),
            n_val=len(self.val_dataset),
            n_test=len(self.test_dataset),
        )

    @property
    def train_dataset(self) -> TACFDataset:
        return self._datasets["train"]

    @property
    def val_dataset(self) -> TACFDataset:
        return self._datasets["val"]

    @property
    def test_dataset(self) -> TACFDataset:
        return self._datasets["test"]

    def _loader(self, split: str, shuffle_override: Optional[bool]) -> DataLoader:
        ds = self._datasets[split]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(
                self.shuffle if shuffle_override is None else bool(shuffle_override)
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and split == "train",
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", None)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)

    def dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def inverse_transform(self, arr: torch.Tensor) -> torch.Tensor:
        """Apply the fitted StandardScaler inverse-transform on the feature axis.

        Parameters
        ----------
        arr : torch.Tensor
            Tensor of arbitrary leading shape ``(..., n_features)``.
        """
        if self.scaler is None:
            return arr
        arr_np = arr.detach().cpu().numpy()
        flat = arr_np.reshape(-1, arr_np.shape[-1])
        inv = self.scaler.inverse_transform(flat).reshape(arr_np.shape)
        return torch.from_numpy(inv.astype(np.float32)).to(arr.device)

    def __repr__(self) -> str:
        parts = [f"TACFDataModule(dataset={self.dataset_name!r})"]
        for split in SPLIT_NAMES:
            ds = self._datasets[split]
            parts.append(f"  {split:>5s}: N={len(ds)}")
        parts.append(f"  n_features={self.n_features}, stamp_dim={self.stamp_dim}")
        return "\n".join(parts)


def build_dataloaders(
    root: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    label_len: int = 0,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_time_features: bool = True,
) -> Tuple[TACFDataModule, DataLoader, DataLoader, DataLoader]:
    """Small helper that constructs a :class:`TACFDataModule` and returns the three loaders."""
    dm = TACFDataModule(
        root=root,
        dataset=dataset,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_time_features=use_time_features,
    )
    return dm, *dm.dataloaders()
