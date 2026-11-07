import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.timefeatures import time_features


Args = namedtuple('Args', ['augmentation_ratio'], defaults=[0])


DATASET_CONFIG = {
    'ETTh1': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTh1.csv',
        'type': 'ett_hour',
        'freq': 'h',
        'target': 'OT',
    },
    'ETTh2': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTh2.csv',
        'type': 'ett_hour',
        'freq': 'h',
        'target': 'OT',
    },
    'ETTm1': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTm1.csv',
        'type': 'ett_minute',
        'freq': 't',
        'target': 'OT',
    },
    'ETTm2': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTm2.csv',
        'type': 'ett_minute',
        'freq': 't',
        'target': 'OT',
    },
    'weather': {
        'root_path': 'dataset/weather',
        'data_path': 'weather.csv',
        'type': 'custom',
        'freq': 't',
        'target': 'OT',
    },
    'exchange_rate': {
        'root_path': 'dataset/exchange_rate',
        'data_path': 'exchange_rate.csv',
        'type': 'custom',
        'freq': 'd',
        'target': 'OT',
    },
    'electricity': {
        'root_path': 'dataset/electricity',
        'data_path': 'electricity.csv',
        'type': 'custom',
        'freq': 'h',
        'target': 'OT',
    },
    'traffic': {
        'root_path': 'dataset/traffic',
        'data_path': 'traffic.csv',
        'type': 'custom',
        'freq': 'h',
        'target': 'OT',
    },
}

OUTPUT_ROOT = 'dataset_processed'


def get_borders_ett_hour(n, seq_len):
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


def get_borders_ett_minute(n, seq_len):
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


def get_borders_custom(n, seq_len):
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]
    return border1s, border2s


def get_time_stamp(df_stamp, freq, timeenc):
    df_stamp = df_stamp.copy()
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.dt.month
        df_stamp['day'] = df_stamp.date.dt.day
        df_stamp['weekday'] = df_stamp.date.dt.weekday
        df_stamp['hour'] = df_stamp.date.dt.hour
        data_stamp = df_stamp.drop(columns=['date']).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp


def process_single_dataset(name, cfg, seq_len_default=336):
    args = Args(augmentation_ratio=0)

    root_path = cfg['root_path']
    data_path = cfg['data_path']
    ds_type = cfg['type']
    freq = cfg['freq']
    target = cfg['target']

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    local_fp = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(local_fp)
    n = len(df_raw)
    print(f"  Total rows: {n}")
    print(f"  Columns: {len(df_raw.columns)} (date + {len(df_raw.columns)-1} features)")
    print(f"  Date range: {df_raw.iloc[0, 0]} ~ {df_raw.iloc[-1, 0]}")

    cols = list(df_raw.columns)
    if 'date' in cols and target in cols:
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]

    seq_len = seq_len_default
    label_len_default = seq_len // 2
    pred_len_default = seq_len // 2

    if ds_type == 'ett_hour':
        border1s, border2s = get_borders_ett_hour(n, seq_len)
    elif ds_type == 'ett_minute':
        border1s, border2s = get_borders_ett_minute(n, seq_len)
    else:
        border1s, border2s = get_borders_custom(n, seq_len)

    splits = ['train', 'val', 'test']
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    scaler = StandardScaler()
    train_border1, train_border2 = border1s[0], border2s[0]
    train_data_raw = df_data[train_border1:train_border2]
    scaler.fit(train_data_raw.values)
    data_scaled = scaler.transform(df_data.values)

    feature_names = list(cols_data)

    result = {
        'meta': {
            'name': name,
            'freq': freq,
            'target': target,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'seq_len_default': seq_len,
            'label_len_default': label_len_default,
            'pred_len_default': pred_len_default,
            'split_ratios': None,
        },
        'splits': {},
    }

    for i, split in enumerate(splits):
        b1 = border1s[i]
        b2 = border2s[i]
        split_len = b2 - b1

        split_data = data_scaled[b1:b2]
        df_stamp = df_raw[['date']][b1:b2]
        stamp_timeenc0 = get_time_stamp(df_stamp, freq, timeenc=0)
        stamp_timeenc1 = get_time_stamp(df_stamp, freq, timeenc=1)

        print(f"  {split:>5s}: [{b1:6d}, {b2:6d}) -> len={split_len:6d}  "
              f"data_shape={split_data.shape}  "
              f"stamp0_shape={stamp_timeenc0.shape}  "
              f"stamp1_shape={stamp_timeenc1.shape}")

        result['splits'][split] = {
            'border': (int(b1), int(b2)),
            'data': split_data.astype(np.float32),
            'time_stamp_discrete': stamp_timeenc0.astype(np.int32),
            'time_stamp_continuous': stamp_timeenc1.astype(np.float32),
            'date': df_stamp['date'].astype(str).values,
        }

    if ds_type == 'custom':
        result['meta']['split_ratios'] = {'train': 0.7, 'val': 0.1, 'test': 0.2}
    else:
        total = border2s[2]
        result['meta']['split_ratios'] = {
            'train': border2s[0] / total,
            'val': (border2s[1] - border2s[0]) / total,
            'test': (border2s[2] - border2s[1]) / total,
        }

    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to: {scaler_path}")

    for split in splits:
        split_info = result['splits'][split]
        np.savez_compressed(
            os.path.join(out_dir, f'{split}.npz'),
            data=split_info['data'],
            time_stamp_discrete=split_info['time_stamp_discrete'],
            time_stamp_continuous=split_info['time_stamp_continuous'],
            date=split_info['date'],
            border=np.array(split_info['border'], dtype=np.int64),
        )
        print(f"  Saved: {out_dir}/{split}.npz  [{split_info['data'].nbytes/1024/1024:.2f} MB raw]")

    meta = result['meta'].copy()
    meta['scaler'] = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
    }
    meta['splits'] = {
        s: {'border': result['splits'][s]['border'], 'length': len(result['splits'][s]['date'])}
        for s in splits
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Meta saved to: {out_dir}/meta.json")

    return result


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    summary = {}
    for name, cfg in DATASET_CONFIG.items():
        try:
            r = process_single_dataset(name, cfg)
            summary[name] = {
                'n_features': r['meta']['n_features'],
                'splits': {
                    s: len(r['splits'][s]['date']) for s in ['train', 'val', 'test']
                },
                'freq': r['meta']['freq'],
                'target': r['meta']['target'],
            }
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = {'error': str(e)}

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, info in summary.items():
        if 'error' in info:
            print(f"  {name:16s}: FAILED - {info['error']}")
        else:
            train_n = info['splits']['train']
            val_n = info['splits']['val']
            test_n = info['splits']['test']
            total = train_n + val_n + test_n
            print(f"  {name:16s}: feat={info['n_features']:3d}  "
                  f"train={train_n:6d}  val={val_n:6d}  test={test_n:6d}  "
                  f"total={total:6d}  freq={info['freq']}")

    with open(os.path.join(OUTPUT_ROOT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary saved to: {OUTPUT_ROOT}/summary.json")


if __name__ == '__main__':
    main()
