import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns 
import os

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
    eps = 1.0e-9
    # Ensure inputs are in the correct format for calculation
    y = y.flatten()
    y_hat = y_hat.flatten()
    
    # Prevents log(0) and log(1-0)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# --- Forward Pass and Loss Functions for Testing ---

# Forward pass (used to calculate loss and predictions on test set)
def forward_pass(X, W1, b1, W2, b2):
    # Transposition to align with the expected format of b1 and b2 (column-vectors)
    Z1 = X.dot(W1.T) + b1.T
    A1 = tanh(Z1)
    Z2 = A1.dot(W2.T) + b2.T
    A2 = sigmoid(Z2)
    return A2

# Calculation of loss on the dataset (X, y)
def calculate_loss(X, y, W1, b1, W2, b2):
    A2 = forward_pass(X, W1, b1, W2, b2)
    return binary_cross_entropy(y, A2)


# Data loading and preparation
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, '..', '..', 'data', 'processed_airline_passenger_satisfaction.csv')
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}. Using dummy data.")
    # Fallback: Create a dummy dataset if the file is not found
    data = {'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'satisfaction_satisfied': np.random.randint(0, 2, 100)}
    df = pd.DataFrame(data)

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

# --- Early Stopping Hyperparameters ---
patience = 5
min_delta = 0.0001
best_loss = np.inf
patience_counter = 0

# Parameter initialization (Xavier for tanh)
W1 = init_rng.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = init_rng.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
b2 = np.zeros((output_dim, 1))

# --- Variables to store the BEST weights and bias (for Early Stopping) ---
best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()


# Training loop using true SGD (update per sample)
n_train = X_train.shape[0]
train_losses = []
test_losses = [] # To store validation loss
test_accuracies = [] # To store validation accuracy per epoch

print("\n--- Starting SGD Training (with Early Stopping) ---")
for epoch in range(epochs):
    perm = rng.permutation(n_train)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    total_train_loss = 0.0

    # Training (Forward and Backward per sample)
    for i in range(n_train):
        x_i = X_shuffled[i].reshape(1, -1)
        y_i = y_shuffled[i].reshape(1, 1)

        # Forward Pass
        Z1 = x_i.dot(W1.T) + b1.T
        A1 = tanh(Z1)
        Z2 = A1.dot(W2.T) + b2.T
        A2 = sigmoid(Z2)

        # Loss Calculation and Accumulation
        loss_i = binary_cross_entropy(y_i, A2)
        total_train_loss += loss_i

        # Backward Pass (Gradient Calculation)
        dZ2 = A2 - y_i 
        dW2 = dZ2.T.dot(A1)
        db2 = dZ2.T
        dA1 = dZ2.dot(W2)
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = dZ1.T.dot(x_i)
        db1 = dZ1.T

        # Parameter Update (SGD)
        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1

    # Epoch Loss Evaluation
    avg_epoch_loss = total_train_loss / n_train
    train_losses.append(avg_epoch_loss)

    # Calculation of Loss and Accuracy on the Test Set (Validation)
    A2_test_temp = forward_pass(X_test, W1, b1, W2, b2)
    avg_test_loss = binary_cross_entropy(y_test, A2_test_temp)
    test_losses.append(avg_test_loss)
    
    y_pred_test_temp = (A2_test_temp > 0.5).astype(int)
    accuracy_test_temp = accuracy_score(y_test, y_pred_test_temp)
    test_accuracies.append(accuracy_test_temp)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss = {avg_epoch_loss:.6f} | Test Loss = {avg_test_loss:.6f} | Test Acc = {accuracy_test_temp:.4f}")

    # --- Early Stopping Check ---
    if avg_test_loss < best_loss - min_delta:
        best_loss = avg_test_loss
        patience_counter = 0
        # Save the current best model weights
        best_W1, best_b1 = W1.copy(), b1.copy()
        best_W2, best_b2 = W2.copy(), b2.copy()
        best_epoch = epoch + 1
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (Patience = {patience}). Best epoch was {best_epoch} with loss {best_loss:.6f}.")
            break

# --- End of Training and Final Evaluation ---

# Use the best parameters found during training (saved before overfitting started)
W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2

print("\n--- Final Evaluation on the Test Set (Using Best Model) ---")

# Final Forward Pass
A2_test = forward_pass(X_test, W1, b1, W2, b2)
y_pred = (A2_test > 0.5).astype(int)

# 8. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, A2_test) 

print("-" * 30)
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("-" * 30)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 7. Error Curves and Visualization
plt.figure(figsize=(10, 5))

# Plot 1: Loss vs. Epochs
# We only plot up to the last executed epoch
plt.plot(train_losses[:len(train_losses)], label='Training Loss')
plt.plot(test_losses[:len(test_losses)], label='Test (Validation) Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Model ({best_epoch})')
plt.title("Loss vs. Epochs (SGD per-sample with Early Stopping)")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss_curves.png') 
plt.close() 

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0 (Dissatisfied)', 'Predicted 1 (Satisfied)'],
            yticklabels=['Actual 0 (Dissatisfied)', 'Actual 1 (Satisfied)'])
plt.title("Confusion Matrix (Early Stop Model)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot 3: Final Metrics Table
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
}
metrics_df = pd.DataFrame(metrics)

fig, ax = plt.subplots(figsize=(6, 2)) # Size adjustment for the table
ax.axis('off') # Removes axes
ax.axis('tight')
table = ax.table(cellText=metrics_df.values,
                     colLabels=metrics_df.columns,
                     cellLoc = 'center', 
                     loc = 'center',
                     colColours=['#f3f3f3']*2)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title("Final Classification Metrics (Early Stop Model)")
plt.savefig('metrics_table.png', bbox_inches='tight', pad_inches=0.1)
plt.close()

print("Plots saved (loss_curves.png, confusion_matrix.png, and metrics_table.png).")