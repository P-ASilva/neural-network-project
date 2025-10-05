# Report: Airline Passenger Satisfaction Prediction

## 1. Dataset Selection

### Dataset: Airline Passenger Satisfaction

Source: Kaggle

URL: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Size: 129,880 passengers x 24 features.

### Reason: 

This dataset presents a highly relevant business problem—predicting customer satisfaction—with sufficient size and complexity (22 input features) to make an MLP model meaningful.

## 2. Dataset Explanation

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