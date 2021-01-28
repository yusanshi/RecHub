from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np
from multiprocessing import Pool


def recall(y_trues, y_scores, k):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1][:, :k]
    return np.mean(
        np.sum(np.take_along_axis(y_trues, orders, axis=-1), axis=-1) /
        np.sum(y_trues, axis=-1))


def mrr(y_trues, y_scores):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1]
    y_trues = np.take_along_axis(y_trues, orders, axis=-1)
    rr_scores = y_trues / (np.arange(y_trues.shape[1]) + 1)
    return np.mean(np.sum(rr_scores, axis=-1) / np.sum(y_trues, axis=-1))


def fast_roc_auc_score(y_trues, y_scores):
    # TODO can it be faster?
    with Pool() as pool:
        return np.mean(pool.starmap(roc_auc_score, zip(y_trues, y_scores)))
