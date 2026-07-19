# =====================================================
# TASK 1: BASIC DATA EXPLORATION
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_excel('HousePrediction.xlsx')

train = df[df['SalePrice'].notnull()].copy()
test = df[df['SalePrice'].isnull()].copy()

print("Full dataset shape:", df.shape)
print("Train dataset shape (has SalePrice):", train.shape)
print("Test dataset shape (no SalePrice):", test.shape)

print("\nFirst 100 rows of train data:")
print(train.head(100))

print("\n--- Column Info ---")
print(train.info())

print("\n--- Column Data Types ---")
print(train.dtypes)

print("\n--- Statistical Summary (numerical columns) ---")
print(train.describe())

# Step 6: Quick look at categorical columns - what unique values they have
categorical_cols = train.select_dtypes(include=['object', 'string']).columns
print("\n--- Categorical Columns and their unique values ---")
for col in categorical_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].unique())

print("\nTask 1 completed successfully.")
