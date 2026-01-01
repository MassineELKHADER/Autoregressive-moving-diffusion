from utils.data_utils.real_datasets import CustomDataset

ds_tr = CustomDataset(name="etth", data_root="./data/datasets/ETTh1.csv", window=192, proportion=0.7, period="train")
ds_va = CustomDataset(name="etth", data_root="./data/datasets/ETTh1.csv", window=192, proportion=0.7, period="val")
ds_te = CustomDataset(name="etth", data_root="./data/datasets/ETTh1.csv", window=192, proportion=0.7, period="test", predict_length=96)

print(len(ds_tr), len(ds_va), len(ds_te))
x_tr = ds_tr[0]
x_te, m_te = ds_te[0]
print(x_tr.shape, x_te.shape, m_te.shape)
