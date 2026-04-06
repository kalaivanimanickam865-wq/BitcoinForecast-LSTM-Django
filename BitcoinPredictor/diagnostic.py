import numpy as np, joblib

y_test       = np.load("data/y_test_v2.npy")
close_scaler = joblib.load("data/close_scaler.pkl")

print("y_test sample  :", y_test[:5])
print("y_test min/max :", y_test.min(), y_test.max())

a = close_scaler.inverse_transform(y_test[:5].reshape(-1, 1))
print("Via close_scaler:", a.flatten())
