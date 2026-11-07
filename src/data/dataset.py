from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAMES = ("train", "val", "test")
AGENT_NAMES = ("trend", "cycle", "local")


@dataclass
class DataSample:
    x: torch.Tensor
    x_stamp: torch.Tensor
    y: torch.Tensor
    y_stamp: torch.Tensor


class TACFDataset(Dataset):
    """Sliding-window multivariate time-series dataset.

    Reads preprocessed outputs stored in ``<root>/<dataset>/{split}.npz`` along
    with ``scaler.pkl`` and ``meta.json``.  Returns (x, x_stamp, y, y_stamp)
    produced with a unit-stride sliding window.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        split: str,
        seq_len: int,
        pred_len: int,
        label_len: int = 0,
        use_time_features: bool = True,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")

        self.root = root
        self.dataset = dataset
        self.split = split
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.label_len = int(label_len)
        self.use_time_features = bool(use_time_features)

        self.folder = os.path.join(self.root, self.dataset)
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"dataset folder not found: {self.folder}")

        npz_path = os.path.join(self.folder, f"{split}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"missing split file: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            self.data = torch.from_numpy(data["data"].astype(np.float32))
            if self.use_time_features:
                if "time_stamp_continuous" in data.files:
                    self.stamp = torch.from_numpy(
                        data["time_stamp_continuous"].astype(np.float32)
                    )
                else:
                    self.stamp = torch.zeros(self.data.shape[0], 0, dtype=torch.float32)
            else:
                self.stamp = torch.zeros(self.data.shape[0], 0, dtype=torch.float32)

        meta_path = os.path.join(self.folder, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        scaler_path = os.path.join(self.folder, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        self.n_samples: int = max(
            0, self.data.shape[0] - self.seq_len - self.pred_len + 1
        )

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])

    @property
    def stamp_dim(self) -> int:
        return int(self.stamp.shape[1])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_start = int(idx)
        s_end = s_start + self.seq_len
        p_end = s_end + self.pred_len

        x = self.data[s_start:s_end]
        y = self.data[s_end:p_end]

        if self.stamp.shape[1] > 0:
            x_stamp = self.stamp[s_start:s_end]
            y_stamp = self.stamp[s_end:p_end]
        else:
            x_stamp = torch.zeros(self.seq_len, 0, dtype=torch.float32)
            y_stamp = torch.zeros(self.pred_len, 0, dtype=torch.float32)

        return x, x_stamp, y, y_stamp

    def __repr__(self) -> str:
        return (
            f"TACFDataset(dataset={self.dataset!r}, split={self.split!r}, "
            f"N={len(self)}, seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"n_features={self.n_features}, stamp_dim={self.stamp_dim})"
        )
