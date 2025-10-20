import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Activation functions
def tanh(x): return np.tanh(x)
def tanh_derivative(z): return 1.0 - np.tanh(z) ** 2
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def mse_loss_per_sample(y_true, y_pred): return 0.5 * np.mean((y_true - y_pred) ** 2)

def forward_pass(X, W1, b1, W2, b2):
    Z1 = X.dot(W1.T) + b1.T
    A1 = tanh(Z1)
    Z2 = A1.dot(W2.T) + b2.T
    return A1, Z2

# ---------------------- Data Loading ----------------------
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, '..', '..', 'data', 'processed_vehicles.csv')
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}. Using dummy data.")
    data = {'feature1': np.random.rand(200), 'feature2': np.random.rand(200), 'selling_price': np.random.rand(200)*1e5}
    df = pd.DataFrame(data)

TARGET = 'selling_price'
if "id" in df.columns:
    df = df.drop(columns=["id"])

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].to_numpy(dtype=float)
y = df[[TARGET]].to_numpy(dtype=float)

# ---------------------- Split: 85% train/val + 15% test ----------------------
rng = np.random.RandomState(42)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Data split: train/val={X_trainval.shape[0]} | test={X_test.shape[0]}")

# ---------------------- Data Normalization ----------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_trainval = scaler_X.fit_transform(X_trainval)
X_test = scaler_X.transform(X_test)

y_trainval = scaler_y.fit_transform(y_trainval)
y_test = scaler_y.transform(y_test)

# ---------------------- Baseline Model ----------------------
y_trainval_original = scaler_y.inverse_transform(y_trainval)
mean_baseline = np.mean(y_trainval_original)
y_baseline = np.full_like(y_test, mean_baseline)

# ---------------------- Hyperparameters ----------------------
input_dim = X.shape[1]
hidden_dim = 32
output_dim = 1
eta = 0.001
epochs = 100
batch_size = 16
patience = 10
min_delta = 1e-6
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------- Output directory ----------------------
output_dir = os.path.join(base_path, 'K-fold')
os.makedirs(output_dir, exist_ok=True)

# ---------------------- K-Fold Training ----------------------
fold_results = []
fold_idx = 1
fold_histories = []

for train_idx, val_idx in kf.split(X_trainval):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    init_rng = np.random.RandomState(42)
    W1 = init_rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((hidden_dim, 1))
    W2 = init_rng.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((output_dim, 1))

    best_loss = np.inf
    patience_counter = 0
    best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()

    train_losses = []
    val_losses = []

    print(f"\n--- Fold {fold_idx} ---")
    for epoch in range(epochs):
        perm = rng.permutation(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]
        total_train_loss = 0.0
        num_batches = 0

        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            xb, yb = X_train[start:end], y_train[start:end]
            current_batch_size = xb.shape[0]

            # Forward pass
            A1, A2 = forward_pass(xb, W1, b1, W2, b2)
            loss = mse_loss_per_sample(yb, A2)
            total_train_loss += loss
            num_batches += 1

            # Backpropagation
            dZ2 = (A2 - yb) / current_batch_size
            dW2 = dZ2.T.dot(A1)
            db2 = np.sum(dZ2, axis=0, keepdims=True).T
            
            dA1 = dZ2.dot(W2)
            dZ1 = dA1 * tanh_derivative(A1)
            dW1 = dZ1.T.dot(xb)
            db1 = np.sum(dZ1, axis=0, keepdims=True).T

            # Update weights
            W1 -= eta * dW1
            b1 -= eta * db1
            W2 -= eta * dW2
            b2 -= eta * db2

        avg_train_loss = total_train_loss / num_batches
        _, val_pred = forward_pass(X_val, W1, b1, W2, b2)
        val_mse = mse_loss(y_val, val_pred)

        train_losses.append(avg_train_loss)
        val_losses.append(val_mse)

        # Early stopping
        if val_mse < best_loss - min_delta:
            best_loss = val_mse
            best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    fold_histories.append((train_losses, val_losses))

    # Evaluate fold
    _, val_pred = forward_pass(X_val, best_W1, best_b1, best_W2, best_b2)
    fold_mse = mean_squared_error(y_val, val_pred)
    fold_mae = mean_absolute_error(y_val, val_pred)
    fold_r2 = r2_score(y_val, val_pred)
    fold_results.append((fold_mse, fold_mae, fold_r2))
    fold_idx += 1

# ---------------------- Loss Curves ----------------------
plt.figure(figsize=(10, 6))
for i, (train_losses, val_losses) in enumerate(fold_histories, 1):
    plt.plot(train_losses, label=f'Fold {i} Train Loss', linestyle='--')
    plt.plot(val_losses, label=f'Fold {i} Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fold_loss_curves.png"))
plt.close()

# Average validation curve
val_curves = [np.array(v) for _, v in fold_histories]
max_len = max(len(v) for v in val_curves)
padded_val_curves = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in val_curves])
avg_val_loss = np.nanmean(padded_val_curves, axis=0)

plt.figure(figsize=(8, 5))
plt.plot(avg_val_loss, label='Average Validation Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Average Validation Loss Across Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_val_loss.png"))
plt.close()

# ---------------------- K-Fold Results ----------------------
avg_mse = np.mean([m for m, _, _ in fold_results])
avg_mae = np.mean([a for _, a, _ in fold_results])
avg_r2 = np.mean([r for _, _, r in fold_results])

print("\n--- K-Fold Validation Results ---")
print(f"Average Val MSE : {avg_mse:.4f}")
print(f"Average Val MAE : {avg_mae:.4f}")
print(f"Average Val R²  : {avg_r2:.4f}")

# ---------------------- Final Model ----------------------
init_rng = np.random.RandomState(42)
W1_final = init_rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
b1_final = np.zeros((hidden_dim, 1))
W2_final = init_rng.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
b2_final = np.zeros((output_dim, 1))

for epoch in range(epochs):
    perm = rng.permutation(X_trainval.shape[0])
    X_trainval, y_trainval = X_trainval[perm], y_trainval[perm]
    
    for start in range(0, X_trainval.shape[0], batch_size):
        end = start + batch_size
        xb, yb = X_trainval[start:end], y_trainval[start:end]
        current_batch_size = xb.shape[0]

        A1, A2 = forward_pass(xb, W1_final, b1_final, W2_final, b2_final)
        
        dZ2 = (A2 - yb) / current_batch_size
        dW2 = dZ2.T.dot(A1)
        db2 = np.sum(dZ2, axis=0, keepdims=True).T
        
        dA1 = dZ2.dot(W2_final)
        dZ1 = dA1 * tanh_derivative(A1)
        dW1 = dZ1.T.dot(xb)
        db1 = np.sum(dZ1, axis=0, keepdims=True).T

        W1_final -= eta * dW1
        b1_final -= eta * db1
        W2_final -= eta * dW2
        b2_final -= eta * db2

# ---------------------- Test Evaluation ----------------------
_, y_pred_test = forward_pass(X_test, W1_final, b1_final, W2_final, b2_final)

# Inverse transform for original scale
y_pred_test_original = scaler_y.inverse_transform(y_pred_test)
y_test_original = scaler_y.inverse_transform(y_test)

# Calculate metrics
mse = mean_squared_error(y_test_original, y_pred_test_original)
mae = mean_absolute_error(y_test_original, y_pred_test_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_test_original)

# Baseline metrics
baseline_mse = mean_squared_error(y_test_original, scaler_y.inverse_transform(y_baseline))
baseline_mae = mean_absolute_error(y_test_original, scaler_y.inverse_transform(y_baseline))
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_test_original, scaler_y.inverse_transform(y_baseline))

print("\n--- Final Test Performance ---")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ---------------------- Results DataFrame ----------------------
results_df = pd.DataFrame({
    'Model': ['MLP', 'Baseline (Mean)'],
    'MSE': [mse, baseline_mse],
    'RMSE': [rmse, baseline_rmse],
    'MAE': [mae, baseline_mae],
    'R2': [r2, baseline_r2]
})

print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

# ---------------------- Residual Plot ----------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test_original, y_pred_test_original, alpha=0.6)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Predicted vs Actual (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plot.png"))
plt.close()

# Save metrics
metrics = pd.DataFrame({
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2": [r2]
})
metrics.to_csv(os.path.join(output_dir, "final_metrics.csv"), index=False)

print(f"\nPlots and metrics saved in: {output_dir}")