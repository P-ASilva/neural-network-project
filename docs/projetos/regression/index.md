# Housing Prices Data Preprocessing & EDA

## 1. Dataset Selection and Overview

### 1.1. Dataset Information

Source: Kaggle

URL: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

Size: 545 samples x 13 features.

### 1.2. Selection Reason

This dataset presents a classic regression problem - predicting house prices - with appropriate complexity (12 input features) to demonstrate the effectiveness of an MLP model in regression problems.

### 1.3. Overview & Features

The dataset contains information about house prices and their characteristics. The goal is regression: predict the price of a house based on its attributes.

Target Variable: price (Numerical)

Input Features:

- Physical Characteristics: area, number of bedrooms, bathrooms, stories, parking.
- Binary Features: main road, guest room, basement, hot water heating, air conditioning, preferred area.
- Property Status: furnishing status (furnished, semi-furnished, unfurnished).

### 1.4. Domain Context

Accurate house price prediction is crucial for the real estate market, helping buyers, sellers, and investors in their decisions. This model can directly identify the key factors that influence property value.

### 1.5. Potential Issues

- Outliers: Numerical features like 'area' and 'price' may have extreme values that need treatment.
- Scale: Numerical features have very different scales (area vs number of bedrooms).
- Encoding: Categorical features need appropriate encoding for the model.

## 2. Data Processing and Analysis

### 2.1. Data Loading and Feature Definition

The dataset is loaded from KaggleHub, and features are categorized based on their data type for targeted preprocessing.

```py
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.preprocessing import MinMaxScaler
# Load Data
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
df = pd.read_csv(os.path.join(path, "Housing.csv"), header=0)
print(f"Dataset path: {path}")

# Feature Definitions
NUMERICAL_FEATURES = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
BINARY_FEATURES = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
NOMINAL_FEATURES = ['furnishingstatus']
TARGET = 'price'

CATEGORICAL_FEATURES = BINARY_FEATURES + NOMINAL_FEATURES
```

### 2.2. Exploratory Data Analysis (EDA)

Initial checks confirm the data structure, types, descriptive statistics, and integrity (missing values/unique categories).

#### 2.2.1. Data Structure and Non-Null Counts (df.info())

The initial check confirms the total number of entries and the data types of all columns.

```py
# DATA STRUCTURE AND NON-NULL COUNTS
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 545 entries, 0 to 544
Data columns (total 13 columns):
 #   Column            Non-Null Count  Dtype 
---  ------            --------------  ----- 
 0   price             545 non-null    int64 
 1   area              545 non-null    int64 
 2   bedrooms          545 non-null    int64 
 3   bathrooms         545 non-null    int64 
 4   stories           545 non-null    int64 
 5   mainroad          545 non-null    object
 6   guestroom         545 non-null    object
 7   basement          545 non-null    object
 8   hotwaterheating   545 non-null    object
 9   airconditioning   545 non-null    object
 10  parking           545 non-null    int64 
 11  prefarea          545 non-null    object
 12  furnishingstatus  545 non-null    object
dtypes: int64(6), object(7)
memory usage: 55.5+ KB
```

#### 2.2.2. Descriptive Statistics (df.describe())

Summary statistics for the defined numerical features show the distribution, center, and spread of the data.

```py
# DESCRIPTIVE STATISTICS (Numerical Features)
           count          mean           std      min       25%       50%       75%        max
area       545.0  5150.541284  2170.141023  1590.0  3600.0  4600.0  6360.0  16200.0
bedrooms   545.0     2.965138     0.738064     1.0     2.0     3.0     3.0      6.0
bathrooms  545.0     1.286239     0.502883     1.0     1.0     1.0     2.0      4.0
stories    545.0     1.805505     0.867492     1.0     1.0     2.0     2.0      4.0
parking    545.0     0.693578     0.861586     0.0     0.0     0.0     1.0      3.0
```

#### 2.2.3. Missing Values Check

The check confirms that the initial dataset contains no missing values, simplifying the cleaning process.

```py
# MISSING VALUES COUNT
price               0
area                0
bedrooms            0
bathrooms           0
stories             0
mainroad            0
guestroom           0
basement            0
hotwaterheating     0
airconditioning     0
parking             0
prefarea            0
furnishingstatus    0
dtype: int64
```

#### 2.2.4. Categorical Feature Analysis (Value Counts)

The value counts for categorical features confirm the distribution and unique values, which is essential before encoding.

```py
# CATEGORICAL FEATURE VALUE COUNTS

--- MAINROAD ---
yes    468
no      77
Name: mainroad, dtype: int64

--- GUESTROOM ---
no     448
yes     97
Name: guestroom, dtype: int64

--- BASEMENT ---
no     354
yes    191
Name: basement, dtype: int64
... (omitted for brevity)

--- FURNISHINGSTATUS ---
semi-furnished    238
unfurnished       178
furnished         129
Name: furnishingstatus, dtype: int64
```

## 3. Data Cleaning and Feature Engineering

The following steps are applied to prepare the data for consumption by a machine learning model.

### 3.1. Duplicates, Missing Values, and Outlier Treatment

- Duplicate Removal: Any redundant rows are dropped.
- Missing Value Imputation: If missing values were present (though not in this specific snapshot), numerical features would be imputed with the median, and categorical features with the mode.
- Outlier Handling: Outliers in the price (Target) and area features are capped at the 99th percentile to mitigate their disproportionate influence on the model.

### 3.2. Feature Encoding and Scaling

The final steps involve converting text-based categorical features into numerical formats and scaling the numerical features.

```py
# Duplicates, Missing Values, and Outlier Treatment
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)

# Handle Missing Values (Imputation logic for robustness)
if df.isnull().sum().any():
    for col in NUMERICAL_FEATURES:
        df[col].fillna(df[col].median(), inplace=True)
    for col in BINARY_FEATURES + NOMINAL_FEATURES:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Outlier Treatment (Capping at 99th percentile for 'price' and 'area')
for col in [TARGET, 'area']:
    upper_bound = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Feature Encoding
# Binary Encoding ('yes'/'no' -> 1/0)
df[BINARY_FEATURES] = df[BINARY_FEATURES].replace({'yes': 1, 'no': 0})

# One-Hot Encoding for Nominal Features (e.g., 'furnishingstatus')
df_encoded = pd.get_dummies(df, columns=NOMINAL_FEATURES, drop_first=True, dtype=int)

# Feature Scaling
features_to_scale = [col for col in df_encoded.columns if col in NUMERICAL_FEATURES and col != TARGET]
scaler = MinMaxScaler()

# Min-Max Scaling on selected numerical features
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

print("\n--- Final Encoded DataFrame Head (Post-Processing) ---")
print(df_encoded.head())
print(f"\nFinal DataFrame Shape: {df_encoded.shape}")
```


## 4. **MLP Implementation**

First, we need to import the relevant libraries:

 - Pandas to read the dataset
 - Numpy to perform matrix operations
 - MatPlotLib to generate graphs
 - Sklearn.metrics to calculate metrics of the regression model
 - 

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
```

Then, we need to define the important functions and derivatives:
```Python
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
```

In this case, Tanh is being used as the activation for the hidden layer, the fact that the data is normalized between -1 and 1 helps using this activation function. Since this is a regression model, the output is linear, and has no activation. For this same reason, we are using MSE (mean squared error) to calculate the loss

Following that, we can load the dataset and select the appropriate feature and target columns as well as define the hyparparemeters that are going to be used for the training.

```Python
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
```

```Python
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

best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
```

This is the entire training script, I will then breakdown each section separately:

```Python
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

```
### 5.1. Forward pass
On the forward pass we perform 4 steps:
 - Compute the weighted sum of inputs for each hidden neuron
 - Apply the activation function (tanh) to the hidden layer 
 - Weighted sum of hidden activations for the single output neuron
 - Produce linear real-valued output (suitable for continuous target)

 This is how those steps are performed in the code:
 ```Python
# Forward
Z1 = x_i.dot(W1.T) + b1.T
A1 = tanh(Z1)
Z2 = A1.dot(W2.T) + b2.T
A2 = Z2
 ```

### 5.2. Calculate Loss
The loss is quickly calculated in the following code:
```Python
# Loss (per-sample)
loss_i = mse_loss_per_sample(y_i, A2)
total_train_loss += loss_i
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
avg_train_loss = total_train_loss / n_train
train_losses.append(avg_train_loss)
```
