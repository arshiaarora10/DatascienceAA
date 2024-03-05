
# Python Script for Comprehensive Analysis of XYZ Sportswear Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = '/mnt/data/XYZ_Sportswear_Dummy_Dataset.csv'
dataset = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
dataset['Order Date'] = pd.to_datetime(dataset['Order Date'], format='%Y-%m-%d', errors='coerce')
dataset['Requested Delivery Date'] = pd.to_datetime(dataset['Requested Delivery Date'], format='%Y-%m-%d', errors='coerce')
dataset.dropna(subset=['Order Date', 'Requested Delivery Date'], inplace=True)
dataset['Value'] = pd.to_numeric(dataset['Value'], errors='coerce')
dataset['Items'] = pd.to_numeric(dataset['Items'], errors='coerce')

# EDA: Distribution of Orders by Country, Product, and Route
sns.countplot(data=dataset, x='Customer Country Code')
plt.title('Orders by Country')
plt.show()

sns.countplot(data=dataset, x='Product Code')
plt.title('Orders by Product')
plt.show()

sns.countplot(data=dataset, x='Route')
plt.title('Orders by Route')
plt.show()

# Aggregate Demand Calculation using SARIMA
dataset['YearMonth'] = dataset['Order Date'].dt.to_period('M')
monthly_demand = dataset.groupby('YearMonth')['Items'].sum()
model = SARIMAX(monthly_demand, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Choice Model for SKU-Level Demand
# One-hot encoding for categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(dataset[['Customer Country Code', 'Product Code', 'Route']])

# Splitting the dataset into training and testing sets for the choice model
X_train, X_test, y_train, y_test = train_test_split(encoded_features, dataset['Items'], test_size=0.3, random_state=42)

# Logistic Regression model for choice probabilities
choice_model = LogisticRegression(max_iter=1000)
choice_model.fit(X_train, y_train)

# Combining SARIMA and Choice Model outputs
# This section needs customization based on specific requirements and data structure

# The script covers data loading, cleaning, EDA, SARIMA for aggregate demand forecasting, and a logistic regression model for SKU-level choice probabilities.

# Combining SARIMA and Choice Model outputs

# Assuming SARIMA model has provided an aggregate forecast for a future period
aggregate_forecast = results.get_forecast(steps=12).predicted_mean

# Predicting choice probabilities using the logistic regression model
# Note: In practice, this might require additional data preprocessing
choice_probabilities = choice_model.predict_proba(X_test)

# Assuming each row in choice_probabilities corresponds to an SKU
# and sum of probabilities across all SKUs equals 1
# The final demand for each SKU is calculated as follows:
sku_forecast = aggregate_forecast[-1] * choice_probabilities.mean(axis=0)

# The resulting 'sku_forecast' contains the estimated demand for each SKU
print("Estimated Demand for Each SKU in the Last Forecast Period:", sku_forecast)
