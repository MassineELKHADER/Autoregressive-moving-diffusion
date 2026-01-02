import torch
from utils.data_utils.M4_dataset import build_m4_dataloaders


train_dl, val_dl, test_dl, pred_len = build_m4_dataloaders(
    m4_folder="./data/datasets/M4",
    freq="Monthly",
    window=192,
    batch_size=8,   # small for inspection
)

print("pred_len =", pred_len)

# ---- TRAIN ----
x_train = next(iter(train_dl))
print("\n[TRAIN]")
print("x_train shape:", x_train.shape)
print("x_train stats:", x_train.mean().item(), x_train.std().item())

# ---- VAL ----
x_val = next(iter(val_dl))
print("\n[VAL]")
print("x_val shape:", x_val.shape)

# ---- TEST ----
x_test, mask = next(iter(test_dl))
print("\n[TEST]")
print("x_test shape:", x_test.shape)
print("mask shape:", mask.shape)

# ---- Mask sanity ----
print("\nMask checks:")
print("mask dtype:", mask.dtype)
print("mask unique values:", torch.unique(mask))
print("masked future timesteps:", (~mask[:, -pred_len:, :]).all().item())
print("past observed:", mask[:, :-pred_len, :].all().item())

# ---- Visual check (single series) ----
i = 0
print("\nExample series (first 10 values):")
print("past:", x_test[i, :10, 0])
print("future (GT):", x_test[i, -pred_len:, 0])
print("future masked:", mask[i, -pred_len:, 0])
