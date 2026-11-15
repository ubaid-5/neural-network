"""
nlufx_ref.py - bit-accurate reference model for the reduced-scale on-device training accelerator.

Follows the project proposal and the base paper (Rostami et al., NLU, IEEE OJCAS 2025):
  * Batch size one. Three primitives only: vector-matrix multiply (VMM, and its transposed
    form VMMT), outer product, element-wise add/sub/mul. Plus ReLU and STEP.
  * Tiled execution: LANES MAC lanes, vector tile = LANES words, matrix tile = LANES x LANES.
    One matrix row of a tile is consumed per cycle (LANES products), as in the paper.
  * 16-bit signed fixed point (Qm.F), products accumulated in a 40-bit accumulator,
    round-to-nearest and saturation on write-back.
  * Host (RISC-V in the paper) does softmax, loss/error and the weight update.

Layout convention: weight matrices are [out, in] so that y = W x  (the proposal's y = W.x).
The reference repo stores W as [in, out], so pass W.T.
"""
import numpy as np

W_BITS, ACC_BITS = 16, 40


class Fmt:
    def __init__(self, frac=10):
        self.F = frac
        self.ONE = 1 << frac
        self.MIN, self.MAX = -(1 << (W_BITS - 1)), (1 << (W_BITS - 1)) - 1
        self.ACC_MIN, self.ACC_MAX = -(1 << (ACC_BITS - 1)), (1 << (ACC_BITS - 1)) - 1

    @property
    def name(self):
        return f"Q{W_BITS - self.F}.{self.F}"

    def quant(self, x):
        return np.clip(np.rint(np.asarray(x, dtype=np.float64) * self.ONE), self.MIN, self.MAX).astype(np.int64)

    def to_float(self, q):
        return np.asarray(q, dtype=np.float64) / self.ONE

    def requant(self, acc):
        acc = np.asarray(acc, dtype=np.int64)
        if np.any(acc < self.ACC_MIN) or np.any(acc > self.ACC_MAX):
            raise OverflowError("40-bit accumulator overflow")
        return np.clip((acc + (1 << (self.F - 1))) >> self.F, self.MIN, self.MAX)


def pad_up(n, L):
    return -(-n // L) * L


# ------------------------------------------------------------------ primitives
def vmm_tiled(fmt, a, B, L):
    """y = B x a, executed tile by tile in hardware order. a:[Ncol], B:[Nrow,Ncol] -> [Nrow]."""
    nrow, ncol = B.shape
    a_p = np.pad(a, (0, pad_up(ncol, L) - ncol))
    B_p = np.pad(B, ((0, pad_up(nrow, L) - nrow), (0, pad_up(ncol, L) - ncol)))
    acc = np.zeros(B_p.shape[0], dtype=np.int64)
    for ti in range(0, B_p.shape[0], L):            # row tiles
        for tj in range(0, B_p.shape[1], L):        # column tiles
            acc[ti:ti + L] += (B_p[ti:ti + L, tj:tj + L] * a_p[None, tj:tj + L]).sum(axis=1)
    return fmt.requant(acc[:nrow])


def vmm(fmt, a, B):
    """Same result as vmm_tiled (integer addition is order independent); fast path."""
    return fmt.requant(B.astype(np.int64) @ a.astype(np.int64))


def vmmt(fmt, a, B):
    """y = B^T x a (transposed multiply, used for the backward pass). a:[Nrow], B:[Nrow,Ncol] -> [Ncol]."""
    return vmm(fmt, a, B.T)


def outer(fmt, v, h):
    return fmt.requant(np.outer(v.astype(np.int64), h.astype(np.int64)))


def ew_mul(fmt, a, b):
    return fmt.requant(a.astype(np.int64) * b.astype(np.int64))


def ew_add(fmt, a, b):                               # a*1.0 + b on the MAC datapath, saturating
    return fmt.requant(a.astype(np.int64) * fmt.ONE + (b.astype(np.int64) << fmt.F))


def ew_sub(fmt, a, b):
    return fmt.requant(a.astype(np.int64) * fmt.ONE - (b.astype(np.int64) << fmt.F))


def relu(fmt, x):
    return np.where(x < 0, 0, x).astype(np.int64)


def step(fmt, x):
    return np.where(x > 0, fmt.ONE, 0).astype(np.int64)


# ------------------------------------------------------------------ cycle bounds
def cycle_bounds(nrow, ncol, L):
    """Lower bounds for one VMM on L lanes with an L-word, single-port memory.
    compute: one matrix row (L products) per cycle.
    memory : per tile step L matrix reads + 1 vector read, plus one result write per row tile."""
    rt, ct = pad_up(nrow, L) // L, pad_up(ncol, L) // L
    return {"compute": rt * ct * L, "memory": rt * ct * (L + 1) + rt}


# ------------------------------------------------------------------ network (2 layers, ReLU)
def forward(fmt, xq, W1q, b1q, W2q, b2q, L=None):
    """x:[in] -> (hidden, logits) in Q. L given: tiled hardware order, else fast path."""
    mm = (lambda a, B: vmm_tiled(fmt, a, B, L)) if L else (lambda a, B: vmm(fmt, a, B))
    h = relu(fmt, ew_add(fmt, mm(xq, W1q), b1q))
    z = ew_add(fmt, mm(h, W2q), b2q)
    return h, z


def train_step(fmt, x, target, P, lr, L=None):
    """One batch-size-one SGD step. P = dict of float master params W1[o,i], b1, W2, b2
    (updated in place). Forward and gradients run on the 'accelerator' primitives;
    softmax, error and the weight update run on the host in float, as in the base paper."""
    q = {k: fmt.quant(v) for k, v in P.items()}
    xq = fmt.quant(x)
    h, z = forward(fmt, xq, q["W1"], q["b1"], q["W2"], q["b2"], L)
    zf = fmt.to_float(z)
    p = np.exp(zf - zf.max()); p /= p.sum()
    e = fmt.quant(p - target)                                   # host -> accelerator
    dW2 = outer(fmt, e, h)                                      # outer product
    dh = ew_mul(fmt, vmmt(fmt, e, q["W2"]), step(fmt, h))       # VMMT, STEP, element-wise multiply
    dW1 = outer(fmt, dh, xq)
    for k, g in (("W2", dW2), ("b2", e), ("W1", dW1), ("b1", dh)):
        P[k] -= lr * fmt.to_float(g)                            # host update
    return -np.log(p[int(np.argmax(target))] + 1e-12)


# ------------------------------------------------------------------ self tests
def self_test():
    rng = np.random.default_rng(0)
    f = Fmt(10)
    half = 1 << (f.F - 1)
    assert f.requant(np.array([half]))[0] == 1 and f.requant(np.array([-half]))[0] == 0
    assert f.requant(np.array([f.MAX * f.ONE * 2]))[0] == f.MAX
    assert ew_add(f, np.array([f.MAX]), np.array([f.MAX]))[0] == f.MAX
    x = np.array([-5, 0, 7])
    assert list(relu(f, x)) == [0, 0, 7] and list(step(f, x)) == [0, 0, f.ONE]

    # tiled order == fast path for 2, 4, 8 lanes, including padding (260 -> 264 for 8 lanes, 4 -> 8)
    for (nrow, ncol) in ((24, 260), (4, 24), (5, 13)):
        a = f.quant(rng.uniform(-1, 1, ncol)); B = f.quant(rng.uniform(-0.3, 0.3, (nrow, ncol)))
        ref = vmm(f, a, B)
        for L in (2, 4, 8):
            assert np.array_equal(vmm_tiled(f, a, B, L), ref), (nrow, ncol, L)
    # transposed multiply
    e = f.quant(rng.uniform(-1, 1, 4)); B = f.quant(rng.uniform(-0.3, 0.3, (4, 24)))
    assert np.array_equal(vmmt(f, e, B), f.requant(B.T.astype(np.int64) @ e.astype(np.int64)))

    # one training step: fixed point vs float, same data
    P = {"W1": rng.normal(0, 0.1, (24, 260)), "b1": np.zeros(24), "W2": rng.normal(0, 0.3, (4, 24)), "b2": np.zeros(4)}
    Pf = {k: v.copy() for k, v in P.items()}
    x = rng.uniform(-1, 1, 260); t = np.eye(4)[2]
    train_step(f, x, t, P, 0.01)
    train_step_float(x, t, Pf, 0.01)
    worst = max(np.abs(P[k] - Pf[k]).max() for k in P)
    assert worst < 5e-3, worst
    print(f"self-tests passed (tiled == fast for 2/4/8 lanes; one train step differs from float by {worst:.1e})")
    for L in (2, 4, 8):
        c1, c2 = cycle_bounds(24, 260, L), cycle_bounds(4, 24, L)
        print(f"  {L} lanes: forward-pass lower bound  compute {c1['compute'] + c2['compute']:5d}  memory {c1['memory'] + c2['memory']:5d} cycles")


def train_step_float(x, target, P, lr):
    h = np.maximum(P["W1"] @ x + P["b1"], 0)
    z = P["W2"] @ h + P["b2"]
    p = np.exp(z - z.max()); p /= p.sum(); e = p - target
    dh = (P["W2"].T @ e) * (h > 0)
    P["W2"] -= lr * np.outer(e, h); P["b2"] -= lr * e
    P["W1"] -= lr * np.outer(dh, x); P["b1"] -= lr * dh


if __name__ == "__main__":
    self_test()