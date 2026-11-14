import json
from pathlib import Path
import numpy as np

# Compact speech-command baseline:
# 13 MFCC coefficients x 101 frames -> 10 temporal bins x 13 = 130 features
# MLP: 130 -> 16 -> 4

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mfcc_features_4class"
SEED = 42
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.01
L2 = 1e-4
PATIENCE = 7
CLASS_NAMES = ["yes", "no", "up", "down"]
N_BINS = 20


def load_split(prefix):
    X = np.load(DATA / f"{prefix}_X_9781_norm.npy").astype(np.float32)
    y = np.load(DATA / f"{prefix}_y_9781.npy").astype(np.int64)
    if X.ndim != 3 or X.shape[1] != 13:
        raise ValueError(f"Expected {prefix} features shaped (N, 13, frames), got {X.shape}")
    # Average the time frames into 10 approximately equal temporal bins.
    chunks = np.array_split(X, N_BINS, axis=2)
    pooled = np.concatenate([chunk.mean(axis=2, keepdims=True) for chunk in chunks], axis=2)
    return pooled.reshape(len(X), -1), y


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def evaluate(X, y, W1, b1, W2, b2):
    hidden = np.maximum(0.0, X @ W1 + b1)
    probs = softmax(hidden @ W2 + b2)
    loss = -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)))
    acc = np.mean(np.argmax(probs, axis=1) == y)
    return float(loss), float(acc)


def main():
    np.random.seed(SEED)
    print("LOADING NORMALIZED 9,781-RECORDING DATASET")
    print("==========================================")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")
    print(f"Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Test:       X={X_test.shape}, y={y_test.shape}")

    input_size, hidden_size, output_size = 260, 24, 4
    rng = np.random.default_rng(SEED)
    W1 = (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    W2 = (rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)).astype(np.float32)
    b2 = np.zeros(output_size, dtype=np.float32)

    params = W1.size + b1.size + W2.size + b2.size
    macs = input_size * hidden_size + hidden_size * output_size
    print("\nMODEL")
    print("=====")
    print("Architecture: 260 -> 24 -> 4")
    print("Temporal pooling: average 101 frames into 20 bins per MFCC coefficient")
    print(f"Trainable parameters: {params}")
    print(f"MACs per recording: {macs}")
    print(f"Epochs={EPOCHS}, batch={BATCH_SIZE}, learning_rate={LEARNING_RATE}")

    best_val_acc = -1.0
    best_epoch = 0
    best_weights = None
    best_val_loss = None
    stale = 0
    n = len(X_train)

    print("\nTRAINING")
    print("========")
    for epoch in range(1, EPOCHS + 1):
        order = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            hidden_pre = xb @ W1 + b1
            hidden = np.maximum(0.0, hidden_pre)
            probs = softmax(hidden @ W2 + b2)

            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)

            dW2 = hidden.T @ probs + L2 * W2
            db2 = np.sum(probs, axis=0)
            dh = probs @ W2.T
            dh[hidden_pre <= 0] = 0.0
            dW1 = xb.T @ dh + L2 * W1
            db1 = np.sum(dh, axis=0)

            W1 -= LEARNING_RATE * dW1
            b1 -= LEARNING_RATE * db1
            W2 -= LEARNING_RATE * dW2
            b2 -= LEARNING_RATE * db2

        train_loss, train_acc = evaluate(X_train, y_train, W1, b1, W2, b2)
        val_loss, val_acc = evaluate(X_val, y_val, W1, b1, W2, b2)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, train acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f}, val acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"\nEarly stopping: validation accuracy did not improve for {PATIENCE} epochs.")
                break

    W1, b1, W2, b2 = best_weights
    test_loss, test_acc = evaluate(X_test, y_test, W1, b1, W2, b2)
    hidden = np.maximum(0.0, X_test @ W1 + b1)
    predictions = np.argmax(hidden @ W2 + b2, axis=1)
    cm = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(y_test, predictions):
        cm[actual, predicted] += 1

    print("\nFINAL TEST RESULT (BEST VALIDATION CHECKPOINT)")
    print("==============================================")
    print(f"Best epoch:       {best_epoch}")
    print(f"Validation acc:   {best_val_acc*100:.2f}%")
    print(f"Test loss:        {test_loss:.4f}")
    print(f"Test accuracy:    {test_acc*100:.2f}%")
    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print(f"Class order: {CLASS_NAMES}")
    print(cm)
    for i, name in enumerate(CLASS_NAMES):
        count = int(cm[i].sum())
        correct = int(cm[i, i])
        print(f"{name:5s}: {correct}/{count} = {100.0*correct/count:.2f}%")

    weights_path = DATA / "software_intermediate_260_24_4.npz"
    results_path = DATA / "software_intermediate_results.json"
    np.savez(weights_path, W1=W1, b1=b1, W2=W2, b2=b2)
    results = {
        "architecture": "260-24-4",
        "input_features": 260,
        "temporal_bins": N_BINS,
        "trainable_parameters": int(params),
        "macs_per_recording": int(macs),
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved weights: {weights_path}")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
