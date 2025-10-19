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
