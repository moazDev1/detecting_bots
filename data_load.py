#-------------------------------------------------------------------------
# AUTHORS: Moaz Ali, Gabriel Alfredo Siguenza
# FILENAME: data_mining_project.py
# SPECIFICATION:  
# FOR: CS 5990 - Advanced Data Mining Final Project
#-------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
path = './data/bots_vs_users.csv'
df = pd.read_csv(path)

# Replace 'Unknown' and empty strings with NaN
df.replace(['Unknown', ''], np.nan, inplace=True)

# Missing values
missing = df.isnull().sum()

# Convert numeric-like columns that may be read as object due to 'Unknown'
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            continue

# Fill missing numeric values with column median
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# For categorical columns, fill missing with mode
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Normalize numeric columns using Min-Max scaling
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Skewed features check
skewed_features = df[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)

# One-hot encode categorical columns to ensure all features are numeric
df = pd.get_dummies(df)

# Remove rows with missing target values
df = df[df['target'].notnull()]

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Save the splits as CSV files
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("\nData successfully cleaned, encoded, split, and saved.")
