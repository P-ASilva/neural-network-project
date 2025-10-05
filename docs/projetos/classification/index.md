# Report: Airline Passenger Satisfaction Prediction

## 1. **Dataset Selection**

### Dataset: Airline Passenger Satisfaction

Source: Kaggle

URL: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Size: 129,880 passengers x 24 features.

### Reason: 

This dataset presents a highly relevant business problem—predicting customer satisfaction—with sufficient size and complexity (22 input features) to make an MLP model meaningful.

## 2. **Dataset Explanation**

### 2.1. Overview & Features

The dataset contains survey results from airline passengers. The goal is a binary classification: predict if a passenger is "satisfied" or "neutral or dissatisfied".

Target Variable: satisfaction (Categorical)

Input Features:

Customer & Travel Context: Customer Type (Loyal/Disloyal), Type of Travel, Class, Flight Distance, Delay times.

Service Ratings (Numerical, 1-5): Key features include Online boarding, Seat comfort, Inflight wifi/service, Food and drink, and On-board service.

### 2.2. Domain Context
Understanding the drivers of passenger satisfaction is critical for customer retention and revenue in the competitive airline industry. This model can directly identify key service areas for improvement.

### 2.3. Potential Issues

Class Imbalance: The target is skewed (55% "neutral/dissatisfied", 45% "satisfied").

Missing Values: A small number of missing values exist in the Arrival Delay column.

Outliers: Numerical features like Delay and Flight Distance may have extreme values that need handling.


## 4. **MLP Implementation**

First, we need to import the relevant libraries:

 - Pandas to read the dataset
 - Numpy to perform matrix operations
 - MatPlotLib to generate graphs

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Then, we need to define the important functions and derivatives:
```Python
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
```

In this case, Tanh is being used as the activation for the hidden layer, the fact that the data is normalized between -1 and 1 helps using this activation function. On the other hand, since we are dealing with a binary classification model, the output layer is using sigmoid, as a way to keep the results as 0 and 1.

Following that, we can load the dataset and select the appropriate feature and target columns as well as define the hyparparemeters that are going to be used for the training.

```Python
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
```

```Python
# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 1
eta = 0.01
epochs = 20
init_rng = np.random.RandomState(42)
```

## 5. **Model Training**

Before training, we need to first split the dataset betwen train/test to allow for evaluation:
```Python
rng = np.random.RandomState(42)
perm = rng.permutation(X.shape[0])
split = int(0.8 * X.shape[0])
train_idx = perm[:split]
test_idx = perm[split:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]
```

The parameter were initialized using an approximate method of the Xavier initialization that scales the weight with the number of inputs:
```Python
# Parameter initialization (Xavier for tanh)
W1 = init_rng.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = init_rng.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
b2 = np.zeros((output_dim, 1))
```

This is the entire training script, I will then breakdown each section separately:

```Python
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

```
### 5.1. Forward pass
On the forward pass we perform 4 steps:
 - Compute the weighted sum of inputs for each hidden neuron
 - Apply the activation function (tanh) to the hidden layer 
 - Weighted sum of hidden activations for the single output neuron
 - Convert the output layer into 0-1 probability (sigmoid)

 This is how those steps are performed in the code:
 ```Python
Z1 = x_i.dot(W1.T) + b1.T
A1 = tanh(Z1)
Z2 = A1.dot(W2.T) + b2.T
A2 = sigmoid(Z2)
 ```

### 5.2. Calculate Loss
The loss is quickly calculated in the following code:
```Python
loss_i = binary_cross_entropy(y_i, A2)
total_loss += loss_i
```
Since we are using SGD (calculating loss for each sample), the loss for each sample is added and then divided by the number of samples so we are left with the overall loss for the epoch.

### 5.3. Backwards Propagation
Now we propagate the errors backwards to allow further updating of the weights, this is performed in 6 steps:
 - Output layer error
 - Gradients for output layer weights and bias
 - Backpropagate error into hidden layer
 - Apply tanh derivative to get hidden layer error
 - Gradients for hidden layer weights and bias
 - Parameter updates (gradient descent)

```Python
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
```

At this point we have to factor in the average loss mentioned earlier:
```Python
avg_epoch_loss = total_loss / n_train
train_losses.append(avg_epoch_loss)
```