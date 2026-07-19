# =====================================================
# TASK 2: DATA CLEANING AND ANALYSIS
# =====================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', None)

os.makedirs('charts', exist_ok=True)

df = pd.read_excel('HousePrediction.xlsx')

train = df[df['SalePrice'].notnull()].copy()
test = df[df['SalePrice'].isnull()].copy()

# Q1) Checking for Missing Values
print("--- Missing values in FULL dataset ---")
print(df.isnull().sum())

# Q2) Features with NaN Values
missing_cols = df.columns[df.isnull().any()]
print("\n--- Columns that contain missing values ---")
print(missing_cols.tolist())

# Q3) Mean SalePrice for Missing vs Present Information
for col in missing_cols:
    if train[col].isnull().sum() > 0:
        mean_present = train[train[col].notnull()]['SalePrice'].mean()
        mean_missing = train[train[col].isnull()]['SalePrice'].mean()
        print(f"{col} -> Mean price (present): {mean_present}, Mean price (missing): {mean_missing}")
    else:
        print(f"{col} -> No missing values in train set, so this comparison isn't possible for this column.")

# Q4) Count of Numerical Features
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
print(f"\nNumber of numerical features: {len(numerical_cols)}")
print("Numerical columns:", numerical_cols.tolist())

# Q5) Print First Five Rows of Numerical Values
print("\n--- First five rows of numerical columns ---")
print(train[numerical_cols].head())

# Q6) Compare Difference Between Year Features and SalePrice
train['RemodAge'] = train['YearRemodAdd'] - train['YearBuilt']  # years after building it got remodeled

print("\n--- YearBuilt vs SalePrice correlation ---")
print(train['YearBuilt'].corr(train['SalePrice']))

print("--- YearRemodAdd vs SalePrice correlation ---")
print(train['YearRemodAdd'].corr(train['SalePrice']))

print("--- RemodAge (gap between build & remodel) vs SalePrice correlation ---")
print(train['RemodAge'].corr(train['SalePrice']))

# scatter plot to visualize YearBuilt vs SalePrice
plt.figure(figsize=(8, 5))
plt.scatter(train['YearBuilt'], train['SalePrice'], alpha=0.4)
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.title('Year Built vs Sale Price')
plt.savefig('charts/yearbuilt_vs_saleprice.png')
plt.close()

# Q7) Relationship Between Discrete Variables and SalePrice
plt.figure(figsize=(8, 5))
sns.boxplot(x='OverallCond', y='SalePrice', data=train)
plt.title('Overall Condition vs Sale Price')
plt.savefig('charts/overallcond_vs_saleprice.png')
plt.close()

print("\n--- Average SalePrice by OverallCond rating ---")
print(train.groupby('OverallCond')['SalePrice'].mean())

# Q8) Relationship Between Continuous Variables and SalePrice
continuous_cols = ['LotArea', 'TotalBsmtSF', 'BsmtFinSF2']

for col in continuous_cols:
    plt.figure(figsize=(8, 5))
    plt.scatter(train[col], train['SalePrice'], alpha=0.4)
    plt.xlabel(col)
    plt.ylabel('Sale Price')
    plt.title(f'{col} vs Sale Price')
    plt.savefig(f'charts/{col}_vs_saleprice.png')
    plt.close()

    print(f"Correlation between {col} and SalePrice: {train[col].corr(train['SalePrice'])}")

# Q9) Histogram Analysis for Continuous Variables
for col in continuous_cols + ['SalePrice']:
    plt.figure(figsize=(8, 5))
    train[col].hist(bins=40)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {col}')
    plt.savefig(f'charts/{col}_histogram.png')
    plt.close()

# Q10) Logarithmic Transformation
train['SalePrice_log'] = np.log1p(train['SalePrice'])
train['LotArea_log'] = np.log1p(train['LotArea'])

plt.figure(figsize=(8, 5))
train['SalePrice_log'].hist(bins=40)
plt.xlabel('Log(SalePrice)')
plt.title('Distribution of SalePrice after Log Transformation')
plt.savefig('charts/saleprice_log_histogram.png')
plt.close()

plt.figure(figsize=(8, 5))
train['LotArea_log'].hist(bins=40)
plt.xlabel('Log(LotArea)')
plt.title('Distribution of LotArea after Log Transformation')
plt.savefig('charts/lotarea_log_histogram.png')
plt.close()

print("\nAll charts saved in the 'charts' folder.")
print("Task 2 completed successfully.")
