import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Official M4 horizons by frequency
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
    Reads an M4 CSV (e.g., Monthly-train.csv) where each row is one series:
      [id, y1, y2, ..., yT]
    Returns:
      ids: list of series ids
      series_list: list of 1D np arrays (float32) with NaNs removed
    """
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    series_list = []
    for i in range(values.shape[0]):
        s = values[i]
        # Remove NaNs
        s = s[~np.isnan(s)]
        series_list.append(s.astype(np.float32))
    return ids, series_list


@dataclass
class M4Data:
    freq: str
    ids: List[str]
    train: List[np.ndarray]  # each is (Ti,)
    test: List[np.ndarray]   # each is (H,)


def load_m4_from_folder(folder: str, freq: str) -> M4Data:
    """
    Expects files:
      {freq}-train.csv and {freq}-test.csv in `folder`
    """
    assert freq in M4_HORIZON, f"Unknown freq={freq}. Choose one of {list(M4_HORIZON.keys())}"

    train_path = os.path.join(folder, f"{freq}-train.csv")
    test_path  = os.path.join(folder, f"{freq}-test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing: {test_path}")

    ids_train, train_list = _read_m4_csv(train_path)
    ids_test,  test_list  = _read_m4_csv(test_path)

    # same ordering / same ids
    if ids_train != ids_test:
        # Still try to align by id
        test_map = {i: s for i, s in zip(ids_test, test_list)}
        aligned_test = []
        missing = []
        for i in ids_train:
            if i not in test_map:
                missing.append(i)
            else:
                aligned_test.append(test_map[i])
        if missing:
            raise ValueError(f"Some train ids missing in test file: {missing[:5]} ...")
        test_list = aligned_test

    # Ensure test horizon matches expected (some series can be shorter if corrupted)
    H = M4_HORIZON[freq]
    for k, s in enumerate(test_list):
        if len(s) != H:
            raise ValueError(f"Series {ids_train[k]} test length {len(s)} != expected horizon {H}")

    return M4Data(freq=freq, ids=ids_train, train=train_list, test=test_list)


class M4Dataset(Dataset):
    """
    ARMD-compatible dataset.

    Output shapes:
      train/val: x -> (window, 1) float32
      test: (x, mask) where:
            x -> (window, 1) float32
            mask -> (window, 1) bool, last pred_len steps False

    Key points:
      - per-series scaling, fit ONLY on that series' train part (no leakage)
      - train/val windows come from TRAIN file only
      - test sample uses last (context + horizon), where future comes from M4 test file
    """
    def __init__(
        self,
        m4: M4Data,
        period: str,                 # "train" | "val" | "test"
        window: int,
        pred_len: Optional[int] = None,
        val_ratio: float = 0.1,      # fraction of TRAIN portion used for validation
        stride: int = 1,
        min_length: int = 50,        # skip too-short series
    ):
        super().__init__()
        assert period in ["train", "val", "test"]
        self.period = period
        self.window = int(window)
        self.pred_len = int(pred_len) if pred_len is not None else M4_HORIZON[m4.freq]
        self.val_ratio = float(val_ratio)
        self.stride = int(stride)
        self.min_length = int(min_length)

        H = self.pred_len
        self.samples: List[np.ndarray] = []
        self.masks: Optional[List[np.ndarray]] = [] if period == "test" else None

        for sid, train_series, test_series in zip(m4.ids, m4.train, m4.test):
            if len(train_series) < max(self.min_length, self.window - H + 1):
                # too short to produce windows
                continue

            # ---- scale per series using TRAIN ONLY (no leakage) ----
            mu = float(np.mean(train_series))
            sigma = float(np.std(train_series) + 1e-8)
            train_scaled = (train_series - mu) / sigma
            test_scaled  = (test_series  - mu) / sigma

            # Split TRAIN into train/val by time (no shuffle)
            T = len(train_scaled)
            val_size = int(np.floor(T * self.val_ratio))
            train_end = T - val_size  # last chunk reserved for val

            if period in ["train", "val"]:
                if period == "train":
                    start_idx = 0
                    end_idx = train_end
                else:  # val
                    start_idx = max(0, train_end - (self.window - 1))  # allow some context
                    end_idx = T

                # Build windows fully inside [start_idx, end_idx)
                # but windows are slices of train_scaled
                n = end_idx - start_idx
                num = max(n - self.window + 1, 0)
                for i in range(0, num, self.stride):
                    s = start_idx + i
                    e = s + self.window
                    x = train_scaled[s:e]
                    if len(x) == self.window:
                        self.samples.append(x.astype(np.float32))

            else:
                # We want x of length window where last H points are the true future.
                full = np.concatenate([train_scaled, test_scaled], axis=0)
                if len(full) < self.window:
                    continue
                x = full[-self.window:].astype(np.float32)

                mask = np.ones((self.window,), dtype=bool)
                mask[-H:] = False
                self.samples.append(x)
                assert self.masks is not None
                self.masks.append(mask)

        if len(self.samples) == 0:
            raise ValueError(f"M4Dataset(period={period}) produced 0 samples. Check window/val_ratio/min_length.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = self.samples[idx]                        # (window,)
        x = torch.from_numpy(x).float().unsqueeze(-1)  # (window, 1)

        if self.period == "test":
            mask = self.masks[idx]                    # (window,)
            mask = torch.from_numpy(mask).bool().unsqueeze(-1)  # (window, 1)
            return x, mask

        return x


def build_m4_dataloaders(
    m4_folder: str,
    freq: str,
    window: int,
    batch_size: int,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    stride: int = 1,
):
    """
    Convenience function returning (train_dl, val_dl, test_dl, pred_len).
    """
    m4 = load_m4_from_folder(m4_folder, freq=freq)
    pred_len = M4_HORIZON[freq]

    train_ds = M4Dataset(m4, period="train", window=window, pred_len=pred_len,
                         val_ratio=val_ratio, stride=stride)
    val_ds   = M4Dataset(m4, period="val",   window=window, pred_len=pred_len,
                         val_ratio=val_ratio, stride=stride)
    test_ds  = M4Dataset(m4, period="test",  window=window, pred_len=pred_len,
                         val_ratio=val_ratio, stride=1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_dl, val_dl, test_dl, pred_len
