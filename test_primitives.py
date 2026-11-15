"""
test_primitives.py - directed + constrained-random verification of the fixed-point reference model.

Every primitive of nlufx_ref.py is checked against an INDEPENDENT implementation written with
plain Python integers (no numpy, no tiling), for 2, 4 and 8 lanes, and against float64 for
numerical error. Prints a pass/fail table. Exit code 0 = everything passed.

Run:  python test_primitives.py            (about a minute)
      python test_primitives.py --cases 500 --frac 10
"""
import argparse
import sys
import numpy as np
import nlufx_ref as R

LANES = (2, 4, 8)


# ----------------------------------------------------------- independent integer reference
def rq(acc, F):
    r = (acc + (1 << (F - 1))) >> F
    return max(-32768, min(32767, r))


def ref_vmm(a, B, F):
    out = []
    for row in B:
        acc = sum(int(x) * int(y) for x, y in zip(row, a))
        assert -(1 << 39) <= acc < (1 << 39), "reference accumulator overflow"
        out.append(rq(acc, F))
    return np.array(out, dtype=np.int64)


def ref_vmmt(a, B, F):
    return ref_vmm(a, [list(col) for col in zip(*B)], F)


def ref_outer(v, h, F):
    return np.array([[rq(int(x) * int(y), F) for y in h] for x in v], dtype=np.int64)


def ref_ew(a, b, F, op):
    one = 1 << F
    f = {"mul": lambda x, y: x * y,
         "add": lambda x, y: x * one + (y << F),
         "sub": lambda x, y: x * one - (y << F)}[op]
    return np.array([rq(f(int(x), int(y)), F) for x, y in zip(a, b)], dtype=np.int64)


# ----------------------------------------------------------- stimulus
def rand_vals(rng, shape, F, mode):
    """mode 'small': values in [-1, 1) (float-comparable); 'full': full 16-bit range with extremes."""
    one = 1 << F
    if mode == "small":
        return np.rint(rng.uniform(-1, 1, shape) * one).astype(np.int64)
    v = rng.integers(-32768, 32768, shape).astype(np.int64)
    pick = rng.random(shape)
    v = np.where(pick < 0.08, -32768, v)
    v = np.where((pick >= 0.08) & (pick < 0.16), 32767, v)
    v = np.where((pick >= 0.16) & (pick < 0.24), 0, v)
    return v


def directed_vmm(F):
    mx, mn = 32767, -32768
    cases = []
    for name, a, B in (
        ("zeros", np.zeros(13), np.zeros((5, 13))),
        ("all max", np.full(13, mx), np.full((5, 13), mx)),
        ("all min", np.full(13, mn), np.full((5, 13), mn)),
        ("max x min", np.full(13, mx), np.full((5, 13), mn)),
        ("alternating", np.array([mx, mn] * 7)[:13], np.tile(np.array([mn, mx] * 7)[:13], (5, 1))),
        ("one element", np.array([1 << F]), np.array([[1 << F], [-(1 << F)]])),
        ("rounding half", np.array([1]), np.array([[1 << (F - 1)], [-(1 << (F - 1))], [(1 << (F - 1)) - 1]])),
        ("padding sizes", np.arange(1, 8), np.arange(1, 7 * 9 + 1).reshape(9, 7)),
    ):
        cases.append((name, a.astype(np.int64), B.astype(np.int64)))
    return cases


# ----------------------------------------------------------- test runner
class Tally:
    def __init__(self):
        self.rows = {}

    def add(self, key, ok, info=""):
        r = self.rows.setdefault(key, [0, 0, []])
        r[0] += 1
        if not ok:
            r[1] += 1
            if len(r[2]) < 3:
                r[2].append(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200, help="random cases per primitive")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    F = a.frac
    fmt = R.Fmt(F)
    rng = np.random.default_rng(a.seed)
    T = Tally()

    # 1) requantise unit: boundaries + random accumulators vs independent function
    half = 1 << (F - 1)
    bounds = [0, 1, -1, half - 1, half, half + 1, -half + 1, -half, -half - 1, (32767 << F), (32767 << F) + half - 1,
              (32767 << F) + half, -(32768 << F), -(32768 << F) - half, -(32768 << F) - half - 1, (1 << 39) - 1, -(1 << 39)]
    rand_acc = rng.integers(-(1 << 38), 1 << 38, 5000).tolist()
    for acc in bounds + rand_acc:
        T.add("requantise (round + saturate)", int(fmt.requant(np.array([acc]))[0]) == rq(acc, F), f"acc={acc}")

    # 2) VMM / VMMT: directed + random, every lane count, tiled vs independent reference
    for L in LANES:
        for name, av, Bv in directed_vmm(F):
            ok = np.array_equal(R.vmm_tiled(fmt, av, Bv, L), ref_vmm(av, Bv, F))
            T.add(f"VMM tiled, {L} lanes (directed)", ok, name)
            ok = np.array_equal(R.vmm_tiled(fmt, np.resize(av, Bv.shape[0]), Bv.T, L),
                                ref_vmmt(np.resize(av, Bv.shape[0]), Bv, F))
            T.add(f"VMMT tiled, {L} lanes (directed)", ok, name)
        for _ in range(a.cases):
            nrow, ncol = int(rng.integers(1, 41)), int(rng.integers(1, 71))
            mode = "small" if rng.random() < 0.5 else "full"
            av, Bv = rand_vals(rng, ncol, F, mode), rand_vals(rng, (nrow, ncol), F, mode)
            T.add(f"VMM tiled, {L} lanes (random)", np.array_equal(R.vmm_tiled(fmt, av, Bv, L), ref_vmm(av, Bv, F)),
                  f"{nrow}x{ncol} {mode}")
            ev = rand_vals(rng, nrow, F, mode)
            got = R.vmm_tiled(fmt, ev, Bv.T.copy(), L)
            T.add(f"VMMT tiled, {L} lanes (random)", np.array_equal(got, ref_vmmt(ev, Bv, F)), f"{nrow}x{ncol} {mode}")

    # 3) fast path equals tiled path (what the other scripts rely on)
    for _ in range(a.cases):
        nrow, ncol = int(rng.integers(1, 41)), int(rng.integers(1, 71))
        av, Bv = rand_vals(rng, ncol, F, "full"), rand_vals(rng, (nrow, ncol), F, "full")
        T.add("fast path == tiled path (2/4/8 lanes)",
              all(np.array_equal(R.vmm(fmt, av, Bv), R.vmm_tiled(fmt, av, Bv, L)) for L in LANES))

    # 4) outer product, element-wise ops, ReLU, STEP (lane independent: one value per element)
    for _ in range(a.cases):
        n, m = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        mode = "small" if rng.random() < 0.5 else "full"
        v, h = rand_vals(rng, n, F, mode), rand_vals(rng, m, F, mode)
        T.add("outer product", np.array_equal(R.outer(fmt, v, h), ref_outer(v, h, F)), f"{n}x{m} {mode}")
        x, y = rand_vals(rng, n, F, mode), rand_vals(rng, n, F, mode)
        for op, fn in (("mul", R.ew_mul), ("add", R.ew_add), ("sub", R.ew_sub)):
            T.add(f"element-wise {op}", np.array_equal(fn(fmt, x, y), ref_ew(x, y, F, op)), f"n={n} {mode}")
        T.add("ReLU", np.array_equal(R.relu(fmt, x), np.maximum(x, 0)))
        T.add("STEP", np.array_equal(R.step(fmt, x), np.where(x > 0, 1 << F, 0)))
    # directed saturation for element-wise ops
    mx, mn = np.array([32767, -32768, 32767]), np.array([32767, -32768, -32768])
    for op, fn in (("add", R.ew_add), ("sub", R.ew_sub), ("mul", R.ew_mul)):
        T.add(f"element-wise {op}", np.array_equal(fn(fmt, mx, mn), ref_ew(mx, mn, F, op)), "saturation corners")

    # 5) numerical error against float64 (values in [-1, 1), error bound from operand rounding)
    one = 1 << F
    worst = 0.0
    for _ in range(a.cases):
        nrow, ncol = int(rng.integers(1, 41)), int(rng.integers(1, 71))
        af, Bf = rng.uniform(-1, 1, ncol), rng.uniform(-1, 1, (nrow, ncol))
        out = fmt.to_float(R.vmm(fmt, fmt.quant(af), fmt.quant(Bf)))
        err = np.abs(out - Bf @ af).max()
        bound = ncol * 0.5 / one * 2.0 + 0.5 / one + ncol * (0.5 / one) ** 2
        worst = max(worst, err / bound)
        T.add("VMM error vs float64 within bound", err <= bound, f"err={err:.5f} bound={bound:.5f}")

    # ------------------------------------------------------- report
    print(f"fixed-point format Q{16 - F}.{F}, seed {a.seed}\n")
    print(f"{'check':46s} {'cases':>7s} {'pass':>7s} {'fail':>6s}")
    total = fails = 0
    for k, (n, f, info) in T.rows.items():
        print(f"{k:46s} {n:7d} {n - f:7d} {f:6d}" + (f"   e.g. {info[0]}" if f else ""))
        total += n
        fails += f
    print(f"\nTOTAL {total} cases, {total - fails} passed, {fails} failed")
    print(f"largest float error / bound: {worst:.2f} (must stay below 1.00)")
    safe = ((1 << 39) - 1) // (1 << 30)
    print(f"worst-case safe dot-product length with a 40-bit accumulator: {safe} elements "
          f"(your layers: 260 and 24)")
    print("ALL PASS" if fails == 0 else "FAILURES PRESENT")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
