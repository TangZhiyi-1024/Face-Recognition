import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cvproj_exc.config import Config
from cvproj_exc.osr_learning import spl_training, mpl_training, UNKNOWN_LABEL


def main():
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    X = df[:, :-1].astype(float)
    y = df[:, -1].astype(int)

    rng = np.random.default_rng(42)
    known_labels = np.unique(y[y != UNKNOWN_LABEL])

    train_idx, test_idx = [], []
    for lb in known_labels:
        idx = np.where(y == lb)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:2])
        test_idx.extend(idx[2:])

    unk_idx = np.where(y == UNKNOWN_LABEL)[0]
    rng.shuffle(unk_idx)
    s = len(unk_idx) // 2
    train_idx.extend(unk_idx[:s])
    test_idx.extend(unk_idx[s:])

    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    def run(name, train_fn):
        pred_fn = train_fn(Xtr, ytr)
        y_pred, y_score = pred_fn(Xte)

        known = yte != UNKNOWN_LABEL
        unk = ~known

        rank1_known = np.mean(y_pred[known] == yte[known])
        unk_reject = np.mean(y_pred[unk] == UNKNOWN_LABEL)
        auc = roc_auc_score(known.astype(int), y_score)

        print(f"{name}: rank1_known={rank1_known:.4f}, unk_reject={unk_reject:.4f}, auc={auc:.4f}")

    run("SPL", spl_training)
    run("MPL", mpl_training)


if __name__ == "__main__":
    main()
