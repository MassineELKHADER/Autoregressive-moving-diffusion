import torch
from torch.utils.data import DataLoader

from utils.data_utils.M4_dataset import M4Dataset, M4_HORIZON

M4_FOLDER = "./data/datasets/M4"
FREQ = "Monthly"
WINDOW = 192
BATCH_SIZE = 8


def inspect_batch(name, batch):
    print(f"\n[{name}]")

    if isinstance(batch, (tuple, list)):
        x, mask = batch
        print("x shape:", x.shape)
        print("mask shape:", mask.shape)
        print("mask dtype:", mask.dtype)
        print("masked future timesteps:", (~mask[-18:]).all().item())
        print("past observed:", mask[:-18].all().item())
    else:
        x = batch
        print("x shape:", x.shape)

    print("x stats:", x.mean().item(), x.std().item())

def main():
    train_ds = M4Dataset(
        m4_folder=M4_FOLDER,
        freq=FREQ,
        period="train",
        window=WINDOW,
    )

    val_ds = M4Dataset(
        m4_folder=M4_FOLDER,
        freq=FREQ,
        period="val",
        window=WINDOW,
    )

    test_ds = M4Dataset(
        m4_folder=M4_FOLDER,
        freq=FREQ,
        period="test",
        window=WINDOW,
    )

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))
    print("Prediction horizon:", M4_HORIZON[FREQ])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    inspect_batch("TRAIN", next(iter(train_dl)))
    inspect_batch("VAL", next(iter(val_dl)))
    inspect_batch("TEST", next(iter(test_dl)))


if __name__ == "__main__":
    main()
