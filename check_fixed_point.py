"""
check_fixed_point.py - run YOUR trained 260-24-4 network through the fixed-point reference model.

Usage (from the folder that holds your files):
  python check_fixed_point.py --weights software_intermediate_260_24_4.npz \
         --testX test_X_9781_norm.npy --testy test_y_9781.npy

It answers the questions of Section 6 of the proposal:
  1. Which fixed-point format keeps accuracy close to the float32 model?
  2. How large do activations and the accumulator get (range and bit width)?
  3. Is the tiled, lane-by-lane execution bit-exact for 2, 4 and 8 lanes?
  4. What are the cycle lower bounds for one forward pass per lane count?
Nothing here changes your float model; it only reads the files.
"""
import argparse
import numpy as np
from nlufx_ref import Fmt, vmm, ew_add, relu, forward, cycle_bounds

N_BINS = 20  # same temporal pooling as train_intermediate_baseline.py


def load_features(path):
    X = np.load(path).astype(np.float32)
    if X.ndim != 3 or X.shape[1] != 13:
        raise SystemExit(f"expected (N, 13, frames), got {X.shape}")
    chunks = np.array_split(X, N_BINS, axis=2)                      # same as your training script
    pooled = np.concatenate([c.mean(axis=2, keepdims=True) for c in chunks], axis=2)
    return pooled.reshape(len(X), -1).astype(np.float64)


def bits_needed(m):
    return int(np.ceil(np.log2(max(int(m), 1) + 1))) + 1           # signed


def fixed_batch(fmt, X, W1, b1, W2, b2):
    """Fast integer forward for the whole test set (same arithmetic as the tiled hardware order).
    Returns also: operand saturations, activation saturations (layer outputs that hit the 16-bit
    limits) and the peak activation magnitude before saturation."""
    Xq, W1q, W2q = fmt.quant(X), fmt.quant(W1), fmt.quant(W2)
    b1q, b2q = fmt.quant(b1), fmt.quant(b2)
    acc1 = Xq @ W1q.T
    r1 = (acc1 + (1 << (fmt.F - 1))) >> fmt.F                      # layer-1 output before saturation
    h = relu(fmt, ew_add(fmt, fmt.requant(acc1), b1q[None, :]))
    acc2 = h @ W2q.T
    r2 = (acc2 + (1 << (fmt.F - 1))) >> fmt.F                      # layer-2 output before saturation
    z = ew_add(fmt, fmt.requant(acc2), b2q[None, :])
    op_sat = int((np.abs(X) * fmt.ONE > fmt.MAX).sum() + (np.abs(W1) * fmt.ONE > fmt.MAX).sum()
                 + (np.abs(W2) * fmt.ONE > fmt.MAX).sum())
    act_sat = int(((r1 > fmt.MAX) | (r1 < fmt.MIN)).sum() + ((r2 > fmt.MAX) | (r2 < fmt.MIN)).sum()
                  + (np.abs(h) >= fmt.MAX).sum() + (np.abs(z) >= fmt.MAX).sum())
    peak = max(np.abs(acc1).max(), np.abs(acc2).max()) / fmt.ONE ** 2 + max(np.abs(b1).max(), np.abs(b2).max())
    return Xq, h, z, max(np.abs(acc1).max(), np.abs(acc2).max()), op_sat, act_sat, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--testX", required=True)
    ap.add_argument("--testy", required=True)
    ap.add_argument("--fracs", type=int, nargs="+", default=[12, 11, 10, 9])
    ap.add_argument("--n-tiled", type=int, default=30, help="samples used for the (slow) tiled bit-exact check")
    a = ap.parse_args()

    w = np.load(a.weights)
    W1, b1, W2, b2 = (w[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))   # repo layout: W1[in,out]
    X, y = load_features(a.testX), np.load(a.testy).astype(np.int64)
    W1t, W2t = W1.T.copy(), W2.T.copy()                                              # -> [out, in]
    print(f"network {W1.shape[0]}-{W1.shape[1]}-{W2.shape[1]}, test samples {len(X)}")

    hf = np.maximum(X @ W1 + b1, 0.0)
    zf = hf @ W2 + b2
    pred_f = zf.argmax(1)
    acc_f = (pred_f == y).mean()
    print(f"\nfloat reference accuracy: {100 * acc_f:.2f}%")
    print(f"float ranges: |x|max {np.abs(X).max():.2f}  |W1|max {np.abs(W1).max():.2f}  |W2|max {np.abs(W2).max():.2f}  "
          f"|hidden|max {hf.max():.2f}  |logits|max {np.abs(zf).max():.2f}")

    print("\nformat | accuracy | change vs float | agrees with float | operand sat. | activation sat. | peak / range | accumulator bits")
    best = None
    for F in a.fracs:
        fmt = Fmt(F)
        Xq, h, z, accmax, op_sat, act_sat, peak = fixed_batch(fmt, X, W1t, b1, W2t, b2)
        pred = z.argmax(1)
        acc = (pred == y).mean()
        agree = (pred == pred_f).mean()
        rng_max = fmt.MAX / fmt.ONE
        print(f"{fmt.name:6s} | {100 * acc:6.2f}%  | {100 * (acc - acc_f):+6.2f} pts      | {100 * agree:6.2f}%           "
              f"| {op_sat:6d}       | {act_sat:6d}          | {peak / rng_max:5.2f}        | {bits_needed(accmax)} (limit 40)")
        if best is None and acc >= acc_f - 0.005 and op_sat == 0 and act_sat == 0 and peak <= 0.5 * rng_max:
            best = fmt.name

    print(f"\nsuggested format: {best or 'none passes all checks: widen the search or retrain'}")
    print("rule: the format with the most fraction bits that has NO saturation (operands or activations), keeps")
    print("      the peak activation at or below half of the format range (headroom for training), and stays")
    print("      within 0.5 points of float accuracy.")

    print(f"\nbit-exact check of tiled execution on {a.n_tiled} samples (2, 4 and 8 lanes vs fast path)")
    fmt = Fmt(a.fracs[min(2, len(a.fracs) - 1)])
    W1q, W2q, b1q, b2q = fmt.quant(W1t), fmt.quant(W2t), fmt.quant(b1), fmt.quant(b2)
    Xq_all = fmt.quant(X[:a.n_tiled])
    for L in (2, 4, 8):
        ok = all(np.array_equal(forward(fmt, Xq_all[i], W1q, b1q, W2q, b2q, L)[1],
                                forward(fmt, Xq_all[i], W1q, b1q, W2q, b2q, None)[1]) for i in range(len(Xq_all)))
        print(f"  {L} lanes: {'bit-exact' if ok else 'MISMATCH'} ({fmt.name})")

    print("\nforward-pass cycle lower bounds (single port, L-word memory, padding included)")
    for L in (2, 4, 8):
        c1, c2 = cycle_bounds(W1t.shape[0], W1t.shape[1], L), cycle_bounds(W2t.shape[0], W2t.shape[1], L)
        print(f"  {L} lanes: compute {c1['compute'] + c2['compute']}  memory {c1['memory'] + c2['memory']}")


if __name__ == "__main__":
    main()
