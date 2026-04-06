import numpy as np
import joblib

# ── Load everything ──────────────────────────────────
y_test_v2    = np.load("data/y_test_v2.npy")
y_test_v1    = np.load("data/y_test.npy")

mf_scaled    = np.load("data/multifeature_predictions_scaled.npy")
hybrid_scaled= np.load("data/hybrid_predictions_scaled.npy")

close_scaler = joblib.load("data/close_scaler.pkl")
old_scaler   = joblib.load("data/scaler.pkl")

print("=" * 55)
print("SHAPE CHECK")
print("=" * 55)
print(f"y_test_v1 (old) shape : {y_test_v1.shape}")
print(f"y_test_v2 (new) shape : {y_test_v2.shape}")
print(f"hybrid_scaled shape   : {hybrid_scaled.shape}")
print(f"mf_scaled shape       : {mf_scaled.shape}")

print("\n" + "=" * 55)
print("RANGE CHECK — scaled values")
print("=" * 55)
print(f"y_test_v1 min/max     : {y_test_v1.min():.4f} / {y_test_v1.max():.4f}")
print(f"y_test_v2 min/max     : {y_test_v2.min():.4f} / {y_test_v2.max():.4f}")
print(f"hybrid_scaled min/max : {hybrid_scaled.min():.4f} / {hybrid_scaled.max():.4f}")
print(f"mf_scaled min/max     : {mf_scaled.min():.4f} / {mf_scaled.max():.4f}")

print("\n" + "=" * 55)
print("USD VALUES CHECK")
print("=" * 55)

actual_v1 = old_scaler.inverse_transform(
    y_test_v1.reshape(-1, 1)).flatten()
actual_v2 = close_scaler.inverse_transform(
    y_test_v2.reshape(-1, 1)).flatten()

hybrid_usd = old_scaler.inverse_transform(
    hybrid_scaled).flatten()
mf_usd = close_scaler.inverse_transform(
    mf_scaled).flatten()

print(f"actual_v1 (old) range : ${actual_v1.min():,.0f} → ${actual_v1.max():,.0f}")
print(f"actual_v2 (new) range : ${actual_v2.min():,.0f} → ${actual_v2.max():,.0f}")
print(f"hybrid_usd range      : ${hybrid_usd.min():,.0f} → ${hybrid_usd.max():,.0f}")
print(f"mf_usd range          : ${mf_usd.min():,.0f} → ${mf_usd.max():,.0f}")

print("\n" + "=" * 55)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("=" * 55)
min_len = min(len(actual_v2), len(mf_usd))
print(f"{'Actual':>12} {'Multi-F Pred':>14} {'Diff':>10}")
print("-" * 40)
for i in range(10):
    diff = mf_usd[i] - actual_v2[i]
    print(f"${actual_v2[i]:>11,.0f} ${mf_usd[i]:>13,.0f} "
          f"${diff:>+10,.0f}")

print("\n" + "=" * 55)
print("TEST PERIOD MATCH CHECK")
print("=" * 55)
print(f"v1 test samples : {len(y_test_v1)}")
print(f"v2 test samples : {len(y_test_v2)}")
print(f"Difference      : {len(y_test_v2) - len(y_test_v1)} samples")
print("(Different window sizes may cause date misalignment)")
