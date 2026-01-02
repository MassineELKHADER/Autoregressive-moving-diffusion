import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Official M4 horizons
M4_HORIZON = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}


def _read_m4_csv(path: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Reads an M4 CSV where each row is:
      [id, y1, y2, ..., yT]
    """
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    series_list = []
    for i in range(values.shape[0]):
        s = values[i]
        s = s[~np.isnan(s)]
        series_list.append(s.astype(np.float32))

    return ids, series_list


@dataclass
class M4Data:
    freq: str
    ids: List[str]
    train: List[np.ndarray]
    test: List[np.ndarray]


def load_m4_from_folder(folder: str, freq: str) -> M4Data:
    assert freq in M4_HORIZON, f"Unknown freq={freq}"

    train_path = os.path.join(folder, f"{freq}-train.csv")
    test_path  = os.path.join(folder, f"{freq}-test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    ids_tr, train_list = _read_m4_csv(train_path)
    ids_te, test_list  = _read_m4_csv(test_path)

    # align by id
    if ids_tr != ids_te:
        test_map = {i: s for i, s in zip(ids_te, test_list)}
        test_list = [test_map[i] for i in ids_tr]

    H = M4_HORIZON[freq]
    for s in test_list:
        if len(s) != H:
            raise ValueError("Invalid M4 test horizon")

    return M4Data(freq=freq, ids=ids_tr, train=train_list, test=test_list)


class M4Dataset(Dataset):
    """
    ARMD-compatible dataset.

    train/val:
      x -> (window, 1)

    test:
      (x, mask)
      x    -> (window, 1)
      mask -> (window, 1), last pred_len = False
    """

    def __init__(
        self,
        m4_folder: str,
        freq: str,
        period: str,           # train | val | test
        window: int,
        val_ratio: float = 0.1,
        stride: int = 1,
        min_length: int = 210,
        **kwargs,
    ):
        super().__init__()
        assert period in ["train", "val", "test"]

        self.period = period
        self.window = int(window)
        self.stride = int(stride)

        m4 = load_m4_from_folder(m4_folder, freq)
        self.pred_len = M4_HORIZON[freq]

        self.samples: List[np.ndarray] = []
        self.masks: Optional[List[np.ndarray]] = [] if period == "test" else None

        for train_series, test_series in zip(m4.train, m4.test):
            if len(train_series) < max(min_length, window - self.pred_len + 1):
                continue

            # ---- scale PER SERIES (train only) ----
            mu = float(train_series.mean())
            sigma = float(train_series.std() + 1e-8)
            train_s = (train_series - mu) / sigma
            test_s  = (test_series  - mu) / sigma

            T = len(train_s)
            val_size = int(T * val_ratio)
            train_end = T - val_size

            if period in ["train", "val"]:
                if period == "train":
                    start, end = 0, train_end
                else:
                    start = max(0, train_end - (window - 1))
                    end = T

                n = end - start
                num = max(n - window + 1, 0)

                for i in range(0, num, self.stride):
                    x = train_s[start + i : start + i + window]
                    if len(x) == window:
                        self.samples.append(x.astype(np.float32))

            else:
                # test
                full = np.concatenate([train_s, test_s], axis=0)
                if len(full) < window:
                    continue

                x = full[-window:].astype(np.float32)
                mask = np.ones((window,), dtype=bool)
                mask[-self.pred_len:] = False

                self.samples.append(x)
                self.masks.append(mask)

        if len(self.samples) == 0:
            raise ValueError("M4Dataset produced 0 samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float().unsqueeze(-1)

        if self.period == "test":
            mask = torch.from_numpy(self.masks[idx]).bool().unsqueeze(-1)
            return x, mask

        return x
