"""
train_batch1_demo.py - batch-size-one training in fixed point vs float, same data, same start.

This is the on-device training scenario of the base paper. Per sample:
  accelerator primitives (fixed point): forward (VMM, bias add, ReLU), error gradient (VMMT),
  STEP, element-wise multiply, outer products.
  host (float): softmax, error = probs - target, weight update (like the RISC-V in the paper).

Run (from the folder that holds nlufx_ref.py and check_fixed_point.py):
  python train_batch1_demo.py --data mfcc_features_4class --epochs 5
Takes a few minutes. Prints validation accuracy after every epoch for both runs, then test accuracy.
"""
import argparse
import time
import numpy as np
import nlufx_ref as R
from check_fixed_point import load_features, fixed_batch


def onehot(c):
    t = np.zeros(4)
    t[c] = 1.0
    return t


def acc_float(P, X, y):
    h = np.maximum(X @ P["W1"].T + P["b1"], 0.0)
    return float(((h @ P["W2"].T + P["b2"]).argmax(1) == y).mean())


def acc_fixed(fmt, P, X, y):
    z = fixed_batch(fmt, X, P["W1"], P["b1"], P["W2"], P["b2"])[2]
    return float((z.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mfcc_features_4class")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lanes", type=int, default=0, help="0 = fast path; 2/4/8 = tiled order (much slower)")
    a = ap.parse_args()
    d = a.data.rstrip("/\\")
    Xtr, ytr = load_features(f"{d}/train_X_9781_norm.npy"), np.load(f"{d}/train_y_9781.npy").astype(np.int64)
    Xva, yva = load_features(f"{d}/validation_X_9781_norm.npy"), np.load(f"{d}/validation_y_9781.npy").astype(np.int64)
    Xte, yte = load_features(f"{d}/test_X_9781_norm.npy"), np.load(f"{d}/test_y_9781.npy").astype(np.int64)
    fmt = R.Fmt(a.frac)
    print(f"train {Xtr.shape}, validation {Xva.shape}, test {Xte.shape}, format {fmt.name}, lr {a.lr}, batch size 1")

    rng = np.random.default_rng(a.seed)
    P0 = {"W1": rng.standard_normal((24, 260)) * np.sqrt(2 / 260), "b1": np.zeros(24),
          "W2": rng.standard_normal((4, 24)) * np.sqrt(2 / 24), "b2": np.zeros(4)}
    Pf = {k: v.copy() for k, v in P0.items()}
    Px = {k: v.copy() for k, v in P0.items()}
    L = a.lanes or None

    print("\nepoch | float val acc | fixed val acc | seconds")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        for i in rng.permutation(len(Xtr)):
            t = onehot(ytr[i])
            R.train_step_float(Xtr[i], t, Pf, a.lr)
            R.train_step(fmt, Xtr[i], t, Px, a.lr, L)
        print(f"{ep:5d} | {100 * acc_float(Pf, Xva, yva):12.2f}% | {100 * acc_fixed(fmt, Px, Xva, yva):12.2f}% | {time.time() - t0:6.0f}")

    af, ax = acc_float(Pf, Xte, yte), acc_fixed(fmt, Px, Xte, yte)
    print(f"\ntest accuracy after {a.epochs} epochs of batch-size-one training")
    print(f"  float   : {100 * af:.2f}%")
    print(f"  {fmt.name} : {100 * ax:.2f}%   (difference {100 * (ax - af):+.2f} points)")
    print("note: weights are kept as float master copies on the host and re-quantised every step.")
    print("      This run uses a single seed; repeat with --seed 1, --seed 2 for a range.")


if __name__ == "__main__":
    main()