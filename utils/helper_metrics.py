import numpy as np

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / (denom + eps))


def extreme_mae(y_true, y_pred, q=0.9):
    thresh = np.quantile(np.abs(y_true), q)
    mask = np.abs(y_true) >= thresh
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))
