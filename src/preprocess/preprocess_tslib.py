#!/usr/bin/env python3
"""TSLib-standard preprocessing for TACF datasets.

Data ROOT (input) : <PROJECT_ROOT>/dataset/   (raw CSV files)
Data ROOT (output): <PROJECT_ROOT>/preprocess/  (per-dataset:
    train.npz / val.npz / test.npz / scaler.pkl / meta.json)

Outputs are byte-for-byte equivalent to the upstream Time-Series-Library
Dataset_ETT_hour / Dataset_ETT_minute / Dataset_Custom classes when
preprocessed with timeenc=0 and timeenc=1 (discrete + continuous time
stamp arrays are saved alongside scaled features).

Usage (in PROJECT_ROOT):
    python -m src.preprocess.preprocess_tslib --all                         # 全部 8 个
    python -m src.preprocess.preprocess_tslib -d ETTh1 -d electricity       # 指定若干
    python -m src.preprocess.preprocess_tslib --list                        # 列出可用数据集
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import namedtuple
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset")
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "preprocess")

Args = namedtuple("Args", ["augmentation_ratio"], defaults=[0])


DATASET_CONFIG = {
    "ETTh1": {
        "root_path": "ETT-small",
        "data_path": "ETTh1.csv",
        "type": "ett_hour",
        "freq": "h",
        "target": "OT",
    },
    "ETTh2": {
        "root_path": "ETT-small",
        "data_path": "ETTh2.csv",
        "type": "ett_hour",
        "freq": "h",
        "target": "OT",
    },
    "ETTm1": {
        "root_path": "ETT-small",
        "data_path": "ETTm1.csv",
        "type": "ett_minute",
        "freq": "t",
        "target": "OT",
    },
    "ETTm2": {
        "root_path": "ETT-small",
        "data_path": "ETTm2.csv",
        "type": "ett_minute",
        "freq": "t",
        "target": "OT",
    },
    "weather": {
        "root_path": "weather",
        "data_path": "weather.csv",
        "type": "custom",
        "freq": "t",
        "target": "OT",
    },
    "exchange_rate": {
        "root_path": "exchange_rate",
        "data_path": "exchange_rate.csv",
        "type": "custom",
        "freq": "d",
        "target": "OT",
    },
    "electricity": {
        "root_path": "electricity",
        "data_path": "electricity.csv",
        "type": "custom",
        "freq": "h",
        "target": "OT",
    },
    "traffic": {
        "root_path": "traffic",
        "data_path": "traffic.csv",
        "type": "custom",
        "freq": "h",
        "target": "OT",
    },
}


# ---------- Time feature encoder (byte-for-byte matches TSLib utils.timefeatures) ----------
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class _TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:  # pragma: no cover - trivial
        raise NotImplementedError


class _SecondOfMinute(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return i.second / 59.0 - 0.5


class _MinuteOfHour(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return i.minute / 59.0 - 0.5


class _HourOfDay(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return i.hour / 23.0 - 0.5


class _DayOfWeek(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return i.dayofweek / 6.0 - 0.5


class _DayOfMonth(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return (i.day - 1) / 30.0 - 0.5


class _DayOfYear(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return (i.dayofyear - 1) / 365.0 - 0.5


class _MonthOfYear(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        return (i.month - 1) / 11.0 - 0.5


class _WeekOfYear(_TimeFeature):
    def __call__(self, i: pd.DatetimeIndex) -> np.ndarray:
        week = i.isocalendar().week.astype(np.int32) if hasattr(i.isocalendar(), "week") else i.week
        return (week - 1) / 52.0 - 0.5


def _time_features_from_freq(freq_str: str) -> List[_TimeFeature]:
    """Identical dispatch to TSLib time_features_from_frequency_str for h/t/d."""
    feats_by = {
        offsets.YearEnd: [],
        offsets.MonthEnd: [_MonthOfYear()],
        offsets.Week:     [_DayOfMonth(), _WeekOfYear()],
        offsets.Day:      [_DayOfWeek(), _DayOfMonth(), _DayOfYear()],
        offsets.BusinessDay: [_DayOfWeek(), _DayOfMonth(), _DayOfYear()],
        offsets.Hour:     [_HourOfDay(), _DayOfWeek(), _DayOfMonth(), _DayOfYear()],
        offsets.Minute:   [_MinuteOfHour(), _HourOfDay(), _DayOfWeek(), _DayOfMonth(), _DayOfYear()],
        offsets.Second:   [_SecondOfMinute(), _MinuteOfHour(), _HourOfDay(),
                           _DayOfWeek(), _DayOfMonth(), _DayOfYear()],
    }
    off = to_offset(freq_str if freq_str not in ("t",) else "min")
    for k_off, v_feat in feats_by.items():
        if isinstance(off, k_off):
            return list(v_feat)
    return feats_by[offsets.Hour]  # fallback


def _time_features_continuous(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Exact TSLib time_features(dates, freq=freq).T with shape (N, D_features)."""
    feats = _time_features_from_freq(freq)
    if not feats:
        return np.zeros((len(dates), 0), dtype=np.float32)
    stacked = np.vstack([f(dates) for f in feats])
    return stacked.T.astype(np.float32)


def _time_features_discrete(dates: pd.DatetimeIndex) -> np.ndarray:
    """timeenc=0: month / day / weekday / hour  (4 columns, matches TSLib default)."""
    month = dates.month.values.astype(np.int32)
    day = dates.day.values.astype(np.int32)
    weekday = dates.dayofweek.values.astype(np.int32)
    hour = dates.hour.values.astype(np.int32)
    return np.stack([month, day, weekday, hour], axis=1).astype(np.int32)


# ---------- Split borders ----------
def get_borders_ett_hour(n: int, seq_len: int):
    _ = n  # ETTh borders are fixed by date, not total length (TSLib rule)
    border1s = [
        0,
        12 * 30 * 24 - seq_len,
        12 * 30 * 24 + 4 * 30 * 24 - seq_len,
    ]
    border2s = [
        12 * 30 * 24,
        12 * 30 * 24 + 4 * 30 * 24,
        12 * 30 * 24 + 8 * 30 * 24,
    ]
    return border1s, border2s


def get_borders_ett_minute(n: int, seq_len: int):
    _ = n
    border1s = [
        0,
        12 * 30 * 24 * 4 - seq_len,
        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
    ]
    border2s = [
        12 * 30 * 24 * 4,
        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
        12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
    ]
    return border1s, border2s


def get_borders_custom(n: int, seq_len: int):
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]
    return border1s, border2s


# ---------- Per-dataset processor ----------
def process_single_dataset(
    name: str,
    cfg: dict,
    seq_len_default: int = 336,
    dataset_root: str = DATASET_ROOT,
    output_root: str = OUTPUT_ROOT,
    overwrite: bool = True,
) -> dict:
    _ = Args(augmentation_ratio=0)

    root_path = cfg["root_path"]
    data_path = cfg["data_path"]
    ds_type = cfg["type"]
    freq = cfg["freq"]
    target = cfg["target"]

    local_root = os.path.join(dataset_root, root_path)
    local_fp = os.path.join(local_root, data_path)
    if not os.path.exists(local_fp):
        raise FileNotFoundError(f"Missing raw CSV: {local_fp}. "
                                f"Place {cfg['data_path']} in dataset/{root_path}/")

    print(f"\n{'=' * 60}")
    print(f"Processing: {name}  (from {local_root})")
    print(f"{'=' * 60}")

    df_raw = pd.read_csv(local_fp)
    n = len(df_raw)
    print(f"  Total rows: {n}")
    print(f"  Columns: {len(df_raw.columns)} (date + {len(df_raw.columns)-1} features)")
    print(f"  Date range: {df_raw.iloc[0, 0]} ~ {df_raw.iloc[-1, 0]}")

    cols = list(df_raw.columns)
    if "date" in cols and target in cols:
        cols.remove(target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [target]]

    seq_len = seq_len_default
    label_len_default = seq_len // 2
    pred_len_default = seq_len // 2

    if ds_type == "ett_hour":
        border1s, border2s = get_borders_ett_hour(n, seq_len)
    elif ds_type == "ett_minute":
        border1s, border2s = get_borders_ett_minute(n, seq_len)
    else:
        border1s, border2s = get_borders_custom(n, seq_len)

    splits = ["train", "val", "test"]
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    scaler = StandardScaler()
    train_border1, train_border2 = border1s[0], border2s[0]
    train_data_raw = df_data[train_border1:train_border2]
    scaler.fit(train_data_raw.values)
    data_scaled = scaler.transform(df_data.values)

    feature_names = list(cols_data)
    result = {
        "meta": {
            "name": name,
            "freq": freq,
            "target": target,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "seq_len_default": seq_len,
            "label_len_default": label_len_default,
            "pred_len_default": pred_len_default,
            "split_ratios": None,
        },
        "splits": {},
    }

    dates = pd.to_datetime(df_raw["date"].values)
    for i, split in enumerate(splits):
        b1 = border1s[i]
        b2 = border2s[i]
        split_len = b2 - b1
        split_dates = dates[b1:b2]

        split_data = data_scaled[b1:b2]
        stamp_disc = _time_features_discrete(split_dates)
        stamp_cont = _time_features_continuous(split_dates, freq)

        print(f"  {split:>5s}: [{b1:6d}, {b2:6d}) -> len={split_len:6d}  "
              f"data_shape={split_data.shape}  "
              f"stamp_disc_shape={stamp_disc.shape}  "
              f"stamp_cont_shape={stamp_cont.shape}")

        result["splits"][split] = {
            "border": (int(b1), int(b2)),
            "data": split_data.astype(np.float32),
            "time_stamp_discrete": stamp_disc.astype(np.int32),
            "time_stamp_continuous": stamp_cont.astype(np.float32),
            "date": split_dates.strftime("%Y-%m-%d %H:%M:%S").astype(str).values,
        }

    if ds_type == "custom":
        result["meta"]["split_ratios"] = {"train": 0.7, "val": 0.1, "test": 0.2}
    else:
        total = border2s[2]
        result["meta"]["split_ratios"] = {
            "train": border2s[0] / total,
            "val": (border2s[1] - border2s[0]) / total,
            "test": (border2s[2] - border2s[1]) / total,
        }

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    def _skip(p: str) -> bool:
        return (not overwrite) and os.path.exists(p)

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    if _skip(scaler_path):
        print(f"  (skip existing) scaler.pkl")
    else:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved to: {scaler_path}")

    for split in splits:
        split_info = result["splits"][split]
        npz_path = os.path.join(out_dir, f"{split}.npz")
        if _skip(npz_path):
            print(f"  (skip existing) {split}.npz")
        else:
            np.savez_compressed(
                npz_path,
                data=split_info["data"],
                time_stamp_discrete=split_info["time_stamp_discrete"],
                time_stamp_continuous=split_info["time_stamp_continuous"],
                date=split_info["date"],
                border=np.array(split_info["border"], dtype=np.int64),
            )
            print(f"  Saved: {npz_path}  "
                  f"[{split_info['data'].nbytes/1024/1024:.2f} MB raw]")

    meta_path = os.path.join(out_dir, "meta.json")
    if _skip(meta_path):
        print(f"  (skip existing) meta.json")
    else:
        meta = result["meta"].copy()
        meta["scaler"] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        }
        meta["splits"] = {
            s: {
                "border": result["splits"][s]["border"],
                "length": len(result["splits"][s]["date"]),
            }
            for s in splits
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Meta saved to: {meta_path}")

    return result


# ---------- CLI ----------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="preprocess_tslib",
        description="Preprocess standard TSLib datasets into TACF-ready .npz files.",
    )
    p.add_argument("-d", "--dataset", action="append", choices=sorted(DATASET_CONFIG.keys()),
                   help="Dataset name, repeat to select multiple. If neither --all nor "
                        "--dataset given, lists available datasets.")
    p.add_argument("--all", action="store_true",
                   help="Preprocess all 8 datasets.")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit.")
    p.add_argument("--seq-len", type=int, default=336,
                   help="Reference seq_len used for ETTh/m border computation.")
    p.add_argument("--overwrite", type=int, default=1, choices=[0, 1],
                   help="1 = overwrite existing outputs, 0 = skip existing files.")
    p.add_argument("--dataset-root", type=str, default=DATASET_ROOT,
                   help=f"Raw CSV root (default: {DATASET_ROOT}).")
    p.add_argument("--output-root", type=str, default=OUTPUT_ROOT,
                   help=f"Preprocessed output root (default: {OUTPUT_ROOT}).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        print("Available datasets:")
        for name, cfg in DATASET_CONFIG.items():
            csv_fp = os.path.join(args.dataset_root, cfg["root_path"], cfg["data_path"])
            status = "OK" if os.path.exists(csv_fp) else "MISSING"
            print(f"  · {name:16s}  type={cfg['type']:11s}  freq={cfg['freq']}  "
                  f"csv={os.path.relpath(csv_fp, _PROJECT_ROOT)}  [{status}]")
        return 0

    if args.all:
        selected: Iterable[str] = DATASET_CONFIG.keys()
    elif args.dataset:
        selected = args.dataset
    else:
        _parse_args(["--help"])
        return 1

    os.makedirs(args.output_root, exist_ok=True)
    print(f"DATASET ROOT  = {args.dataset_root}")
    print(f"OUTPUT ROOT   = {args.output_root}")

    summary: dict = {}
    for name in selected:
        if name not in DATASET_CONFIG:
            print(f"[WARN] Skip unknown dataset: {name}")
            continue
        try:
            r = process_single_dataset(
                name=name, cfg=DATASET_CONFIG[name],
                seq_len_default=args.seq_len,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                overwrite=bool(args.overwrite),
            )
            summary[name] = {
                "n_features": r["meta"]["n_features"],
                "splits": {
                    s: len(r["splits"][s]["date"]) for s in ["train", "val", "test"]
                },
                "freq": r["meta"]["freq"],
                "target": r["meta"]["target"],
            }
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = {"error": str(e)}

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for name, info in summary.items():
        if "error" in info:
            print(f"  {name:16s}: FAILED - {info['error']}")
        else:
            train_n = info["splits"]["train"]
            val_n = info["splits"]["val"]
            test_n = info["splits"]["test"]
            total = train_n + val_n + test_n
            print(f"  {name:16s}: feat={info['n_features']:3d}  "
                  f"train={train_n:6d}  val={val_n:6d}  test={test_n:6d}  "
                  f"total={total:6d}  freq={info['freq']}")

    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary saved to: {args.output_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
