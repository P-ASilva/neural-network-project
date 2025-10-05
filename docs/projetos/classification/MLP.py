import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tanh activation
def tanh(x):
    return np.tanh(x)

# Derivative of tanh with respect to pre-activation z
def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss (mean over batch/sample)
def binary_cross_entropy(y, y_hat):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# Data loading and preparation
csv_path = "./docs/data/processed_airline_passenger_satisfaction.csv"
df = pd.read_csv(csv_path)
target_col = "satisfaction_satisfied"

if "id" in df.columns:
    df = df.drop(columns=["id"])

feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].to_numpy(dtype=float)
y = df[target_col].to_numpy()
y = y.reshape(-1, 1)

rng = np.random.RandomState(42)
perm = rng.permutation(X.shape[0])
split = int(0.8 * X.shape[0])
train_idx = perm[:split]
test_idx = perm[split:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print(f"Samples: total={X.shape[0]}, train={X_train.shape[0]}, test={X_test.shape[0]}")
print(f"Features used: {len(feature_cols)}")

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 1
eta = 0.01
epochs = 20
init_rng = np.random.RandomState(42)

# Parameter initialization (Xavier for tanh)
W1 = init_rng.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = init_rng.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
b2 = np.zeros((output_dim, 1))

# Training loop using true SGD (update per sample)
n_train = X_train.shape[0]
train_losses = []

for epoch in range(epochs):
    perm = rng.permutation(n_train)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    total_loss = 0.0

    for i in range(n_train):
        x_i = X_shuffled[i].reshape(1, -1)
        y_i = y_shuffled[i].reshape(1, 1)

        Z1 = x_i.dot(W1.T) + b1.T
        A1 = tanh(Z1)
        Z2 = A1.dot(W2.T) + b2.T
        A2 = sigmoid(Z2)

        loss_i = binary_cross_entropy(y_i, A2)
        total_loss += loss_i

        dZ2 = A2 - y_i
        dW2 = dZ2.T.dot(A1)
        db2 = dZ2.T
        dA1 = dZ2.dot(W2)
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = dZ1.T.dot(x_i)
        db1 = dZ1.T

        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1

    avg_epoch_loss = total_loss / n_train
    train_losses.append(avg_epoch_loss)

    print(f"Epoch {epoch+1}/{epochs}  Loss = {avg_epoch_loss:.6f}")

# Evaluation on test set
Z1_test = X_test.dot(W1.T) + b1.T
A1_test = tanh(Z1_test)
Z2_test = A1_test.dot(W2.T) + b2.T
A2_test = sigmoid(Z2_test)
y_pred = (A2_test > 0.5).astype(int)

accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot training loss
plt.plot(train_losses)
plt.title("Training Loss (SGD per-sample)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
