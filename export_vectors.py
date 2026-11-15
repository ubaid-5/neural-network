"""
export_vectors.py - write hex test vectors for the RTL testbench from YOUR trained network.

Run (from the folder that holds nlufx_ref.py and check_fixed_point.py):
  python export_vectors.py --weights mfcc_features_4class/software_intermediate_260_24_4.npz ^
        --testX mfcc_features_4class/test_X_9781_norm.npy --testy mfcc_features_4class/test_y_9781.npy ^
        --lanes 4 --frac 10 --n 20 --out vectors

Files written to the --out folder (one 16-bit word per line, 4 hex digits, two's complement):
  W1_tiles.hex  W2_tiles.hex   weight matrices [out, in], padded to multiples of the lane count,
                               stored TILE BY TILE: for each row tile, for each column tile, the
                               LxL tile row by row (one tile row = one bus read of L words).
  *_bus.hex                    the same data packed one tile row (L words) per line, word 0 in the
                               least-significant 16 bits (little-endian bus).
  b1.hex  b2.hex               biases padded to the row padding of their layer.
  x_samples.hex                n input vectors, each padded to the layer-1 column padding.
  h_expected.hex               expected hidden layer (after bias and ReLU), padded.
  z_expected.hex               expected output layer (logits), padded.
  pred_expected.txt            predicted class per sample (argmax of the expected logits).
  labels.txt                   true class per sample.
  manifest.json                sizes, lanes, format, layout description.
Expected values come from the tiled reference model (bit-exact for any lane count).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import nlufx_ref as R
from check_fixed_point import load_features


def hexw(v):
    return f"{int(v) & 0xFFFF:04x}"


def tiles(Mq, L):
    """[rows, cols] int matrix -> flat list of words in tile-by-tile, row-by-row order (padded)."""
    r, c = Mq.shape
    P = np.zeros((R.pad_up(r, L), R.pad_up(c, L)), dtype=np.int64)
    P[:r, :c] = Mq
    out = []
    for ti in range(0, P.shape[0], L):
        for tj in range(0, P.shape[1], L):
            for rr in range(L):
                out.extend(P[ti + rr, tj:tj + L].tolist())
    return out


def write_words(path, words):
    Path(path).write_text("\n".join(hexw(w) for w in words) + "\n")


def write_bus(path, words, L):
    lines = []
    for i in range(0, len(words), L):
        chunk = words[i:i + L]
        lines.append("".join(hexw(w) for w in reversed(chunk)))   # word 0 ends up in the low 16 bits
    Path(path).write_text("\n".join(lines) + "\n")


def pad_vec(v, L):
    out = np.zeros(R.pad_up(len(v), L), dtype=np.int64)
    out[:len(v)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--testX", required=True)
    ap.add_argument("--testy", required=True)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="vectors")
    a = ap.parse_args()
    L = a.lanes
    fmt = R.Fmt(a.frac)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    w = np.load(a.weights)
    W1, b1, W2, b2 = (w[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))
    W1q, W2q = fmt.quant(W1.T), fmt.quant(W2.T)                 # [out, in]
    b1q, b2q = fmt.quant(b1), fmt.quant(b2)
    X = load_features(a.testX)[:a.n]
    y = np.load(a.testy).astype(np.int64)[:a.n]
    Xq = fmt.quant(X)

    for name, Mq in (("W1", W1q), ("W2", W2q)):
        words = tiles(Mq, L)
        write_words(out / f"{name}_tiles.hex", words)
        write_bus(out / f"{name}_tiles_bus.hex", words, L)
    write_words(out / "b1.hex", pad_vec(b1q, L))
    write_words(out / "b2.hex", pad_vec(b2q, L))

    xs, hs, zs, preds = [], [], [], []
    for i in range(len(Xq)):
        h, z = R.forward(fmt, Xq[i], W1q, b1q, W2q, b2q, L)     # tiled hardware order
        xs.extend(pad_vec(Xq[i], L)); hs.extend(pad_vec(h, L)); zs.extend(pad_vec(z, L))
        preds.append(int(np.argmax(z)))
    write_words(out / "x_samples.hex", xs)
    write_words(out / "h_expected.hex", hs)
    write_words(out / "z_expected.hex", zs)
    (out / "pred_expected.txt").write_text("\n".join(map(str, preds)) + "\n")
    (out / "labels.txt").write_text("\n".join(map(str, y.tolist())) + "\n")

    manifest = {
        "format": fmt.name, "frac_bits": a.frac, "lanes": L, "samples": len(Xq),
        "layer1": {"out": W1q.shape[0], "in": W1q.shape[1],
                   "out_padded": R.pad_up(W1q.shape[0], L), "in_padded": R.pad_up(W1q.shape[1], L)},
        "layer2": {"out": W2q.shape[0], "in": W2q.shape[1],
                   "out_padded": R.pad_up(W2q.shape[0], L), "in_padded": R.pad_up(W2q.shape[1], L)},
        "word": "16-bit two's complement, 4 hex digits per line",
        "matrix_layout": "tile by tile (row tile, then column tile), each LxL tile row by row",
        "accumulator_bits": 40,
        "rounding": "round half up (add 2^(F-1), arithmetic shift right F), then saturate to 16 bits",
        "layer_order": "h = ReLU(sat(W1 x) + b1) ; z = sat(W2 h) + b2   (bias added after the requantise)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    agree = np.mean(np.array(preds) == y)
    print(f"wrote {len(list(out.iterdir()))} files to {out}/  ({fmt.name}, {L} lanes, {len(Xq)} samples)")
    print(f"accuracy of these {len(Xq)} samples with the tiled fixed-point model: {100 * agree:.1f}%")


if __name__ == "__main__":
    main()