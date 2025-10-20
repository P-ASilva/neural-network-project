import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Tanh activation
def tanh(x):
    return np.tanh(x)

# Derivative of tanh with respect to pre-activation z
def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

# Regression loss per-sample 0.5*MSE and full-dataset MSE 
def mse_loss(y_true, y_pred):
    # y_true and y_pred are arrays shaped (n_samples, 1)
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_per_sample(y_true, y_pred):
    # use 0.5*(y - yhat)^2 so derivative is (yhat - y)
    return 0.5 * np.mean((y_true - y_pred) ** 2)

# Forward Pass and Loss Functions
def forward_pass(X, W1, b1, W2, b2):
    # X: (n_samples, input_dim)
    Z1 = X.dot(W1.T) + b1.T          
    A1 = tanh(Z1)
    Z2 = A1.dot(W2.T) + b2.T
    A2 = Z2
    return A2

# Data loading and preparation
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, '..', '..', 'data', 'processed_vehicles.csv')
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}. Using dummy data.")
    data = {'feature1': np.random.rand(200), 'feature2': np.random.rand(200), 'selling_price': np.random.rand(200)*1e5}
    df = pd.DataFrame(data)


TARGET = 'selling_price'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in DataFrame.")

# Drop id if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Feature columns - drop any original/backup columns you don't want as features
backup_target_col = f"{TARGET}_original"
feature_cols = [c for c in df.columns if c not in (TARGET, backup_target_col)]

X = df[feature_cols].to_numpy(dtype=float)
y = df[[TARGET]].to_numpy(dtype=float)

# Shuffle and train/test split
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
print(f"Features used: {X.shape[1]}")

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 1
eta = 0.01
epochs = 100
init_rng = np.random.RandomState(42)

# Early stopping params
patience = 10
min_delta = 1e-6
best_loss = np.inf
patience_counter = 0

# Parameter initialization (Xavier for tanh)
W1 = init_rng.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = init_rng.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
b2 = np.zeros((output_dim, 1))

best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()

n_train = X_train.shape[0]
train_losses = []
val_losses = []

print("\n--- Starting SGD Training (regression, per-sample updates) ---")
for epoch in range(epochs):
    perm = rng.permutation(n_train)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    total_train_loss = 0.0

    # SGD per-sample
    for i in range(n_train):
        x_i = X_shuffled[i].reshape(1, -1)
        y_i = y_shuffled[i].reshape(1, 1)

        # Forward
        Z1 = x_i.dot(W1.T) + b1.T
        A1 = tanh(Z1)
        Z2 = A1.dot(W2.T) + b2.T
        A2 = Z2

        # Loss (per-sample)
        loss_i = mse_loss_per_sample(y_i, A2)
        total_train_loss += loss_i

        # Backpropagation
        # dL/dA2 = A2 - y_i
        dZ2 = (A2 - y_i)
        dW2 = dZ2.T.dot(A1)
        db2 = dZ2.T

        dA1 = dZ2.dot(W2)
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = dZ1.T.dot(x_i)
        db1 = dZ1.T

        # Parameter updates
        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1

    avg_train_loss = total_train_loss / n_train
    train_losses.append(avg_train_loss)

    # Validation loss (full test set)
    A2_val = forward_pass(X_test, W1, b1, W2, b2)
    val_mse = mse_loss(y_test, A2_val)
    val_losses.append(val_mse)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss = {avg_train_loss:.8f} | Val MSE = {val_mse:.8f}")

    # Early stopping
    if val_mse < best_loss - min_delta:
        best_loss = val_mse
        patience_counter = 0
        best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
        best_epoch = epoch + 1
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs. Best epoch = {best_epoch} with Val MSE = {best_loss:.8f}")
            break

# Load best params
W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2

# Final predictions (on test set)
A2_test = forward_pass(X_test, W1, b1, W2, b2)  # scaled/log target assumptions apply
y_pred = A2_test.copy()


y_pred_orig = y_pred
y_test_orig = y_test

# Regression metrics
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print("\n--- Final Regression Metrics on Test Set ---")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

# Plot: Loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss (per-sample 0.5*MSE)')
plt.plot(val_losses, label='Validation MSE')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title("Loss vs Epochs (Regression)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curves.png')
plt.close()

# Scatter plot predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Predicted vs Actual (Test Set)')
plt.tight_layout()
plt.savefig('pred_vs_actual.png')
plt.close()

print("Plots saved: loss_curves.png, pred_vs_actual.png")