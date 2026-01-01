import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from model.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.7, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'val', 'test'], "period must be train/val/test."
        if period in ["train", "val"]:
            assert not (predict_length is not None or missing_ratio is not None)
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata = self.read_data(data_root, self.name)

        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]

        self.save2npy = save2npy
        #self.auto_norm = neg_one_to_one
        self.auto_norm = False

        train_ratio = proportion
        val_ratio = 0.1
        train_end, val_end = self.split_time_indices(self.len, train_ratio=train_ratio, val_ratio=val_ratio)
        
        self.scaler = StandardScaler().fit(self.rawdata[:train_end])
        # ---- normalize whole series with train scaler ----
        self.data = self.__normalize(self.rawdata)
        # ---- build windows safely per split ----
        if period == "train":
            start, end = 0, train_end
        elif period == "val":
            start, end = train_end, val_end
        else:  # test
            start, end = val_end, self.len

        self.samples = self.build_windows(self.data, start, end, self.window)
        self.sample_num = self.samples.shape[0]

        if self.period == "test":
            if self.missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif self.pred_len is not None:
                masks = np.ones(self.samples.shape, dtype=bool)
                masks[:, -self.pred_len:, :] = False
                self.masking = masks
            else:
                raise ValueError("For period='test', you must set missing_ratio or predict_length.")

    # def __getsamples(self, data, proportion, seed):
    #     x = np.zeros((self.sample_num_total, self.window, self.var_num))
    #     for i in range(self.sample_num_total):
    #         start = i
    #         end = i + self.window
    #         x[i, :, :] = data[start:end, :]

    #     train_data, test_data = self.divide(x, proportion, seed)

    #     if self.save2npy:
    #         if 1 - proportion > 0:
    #             np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
    #         np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
    #         if self.auto_norm:
    #             if 1 - proportion > 0:
    #                 np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
    #             np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
    #         else:
    #             if 1 - proportion > 0:
    #                 np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
    #             np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

    #     return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    # @staticmethod
    # def divide(data, ratio, seed=2023):
    #     size = data.shape[0]
    #     # Store the state of the RNG to restore later.
    #     st0 = np.random.get_state()
    #     np.random.seed(seed)

    #     regular_train_num = int(np.ceil(size * ratio))
    #     #id_rdm = np.random.permutation(size)
    #     id_rdm = np.arange(size)
    #     regular_train_id = id_rdm[:regular_train_num]
    #     irregular_train_id = id_rdm[regular_train_num:]

    #     regular_data = data[regular_train_id, :]
    #     irregular_data = data[irregular_train_id, :]

    #     # Restore RNG.
    #     np.random.set_state(st0)
    #     return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values.astype(np.float32)
        return data
    
    @staticmethod
    def split_time_indices(n, train_ratio=0.7, val_ratio=0.1):
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return train_end, val_end

    @staticmethod
    def build_windows(data, start_idx, end_idx, window):
        # data is (T, D)
        # we build windows fully contained in [start_idx, end_idx)
        n = end_idx - start_idx
        num = max(n - window + 1, 0)
        x = np.zeros((num, window, data.shape[1]), dtype=np.float32)

        for i in range(num):
            s = start_idx + i
            e = s + window
            x[i] = data[s:e]

        return x

    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

