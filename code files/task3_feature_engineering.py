# =====================================================
# TASK 3: FEATURE ENGINEERING AND ADVANCED ANALYSIS
# =====================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
os.makedirs('charts', exist_ok=True)

df = pd.read_excel('HousePrediction.xlsx')
train = df[df['SalePrice'].notnull()].copy()
test = df[df['SalePrice'].isnull()].copy()

# Q1) Find Outliers
continuous_cols = ['LotArea', 'TotalBsmtSF', 'BsmtFinSF2', 'SalePrice']

for col in continuous_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col} (spotting outliers)')
    plt.savefig(f'charts/{col}_boxplot.png')
    plt.close()

    # IQR method: anything beyond 1.5*IQR from Q1/Q3 is treated as an outlier
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outlier_count = train[(train[col] < lower_limit) | (train[col] > upper_limit)].shape[0]
    print(f"{col}: {outlier_count} outliers found (outside range {lower_limit:.2f} to {upper_limit:.2f})")

# Q2) Relationship Between Categorical Features and SalePrice
categorical_cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

for col in categorical_cols:
    print(f"\n--- Average SalePrice by {col} ---")
    print(train.groupby(col)['SalePrice'].mean().sort_values(ascending=False))

    plt.figure(figsize=(9, 5))
    sns.boxplot(x=col, y='SalePrice', data=train)
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Sale Price')
    plt.tight_layout()
    plt.savefig(f'charts/{col}_vs_saleprice.png')
    plt.close()

# Q3) Correlation Between Numerical Features and SalePrice
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train[numerical_cols].corr()

print("\n--- Correlation of numerical features with SalePrice ---")
print(corr_matrix['SalePrice'].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('charts/correlation_heatmap.png')
plt.close()

# Q4) Continuous Features vs SalePrice 
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, col in enumerate(['LotArea', 'TotalBsmtSF', 'BsmtFinSF2']):
    axes[i].scatter(train[col], train['SalePrice'], alpha=0.4)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('SalePrice')
plt.tight_layout()
plt.savefig('charts/continuous_vs_saleprice_grid.png')
plt.close()

# Q5) Feature Engineering

# Handle Categorical Variables (One-Hot Encoding) 
train_encoded = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
print(f"\nShape before encoding: {train.shape}")
print(f"Shape after one-hot encoding categorical columns: {train_encoded.shape}")

# Handle Numerical Variables (Scaling)
scale_cols = ['LotArea', 'TotalBsmtSF', 'BsmtFinSF2']
scaler = StandardScaler()
train_encoded[scale_cols] = scaler.fit_transform(train_encoded[scale_cols])
print("\nScaled columns preview:")
print(train_encoded[scale_cols].head())

# Handle Temporal Variables
reference_year = train['YearRemodAdd'].max()
train_encoded['HouseAge'] = reference_year - train['YearBuilt']
train_encoded['YearsSinceRemod'] = reference_year - train['YearRemodAdd']
print("\nHouseAge and YearsSinceRemod preview:")
print(train_encoded[['HouseAge', 'YearsSinceRemod']].head())

# ADVANCED QUESTIONS Q11-Q15

# Q11: How does location (zoning + lot config) influence price? 
print("\n--- Q11: Average SalePrice by MSZoning + LotConfig ---")
print(train.groupby(['MSZoning', 'LotConfig'])['SalePrice'].mean().sort_values(ascending=False).head(10))
# Insight: We can see which zoning+lot combinations fetch the highest average price.

# Q12: Impact of overall condition on sale price 
print("\n--- Q12: Average SalePrice by OverallCond (recap from Task 2) ---")
print(train.groupby('OverallCond')['SalePrice'].mean())
# Insight: OverallCond does NOT show a clean straight-line relationship with price.
# Ratings 5 and 9 have unusually high average prices - condition alone is not a strong standalone predictor, it likely interacts with other features.

#  Q13: Does presence of a basement significantly affect price? 
train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 'Yes' if x > 0 else 'No')
print("\n--- Q13: Average SalePrice by basement presence ---")
print(train.groupby('HasBasement')['SalePrice'].mean())

plt.figure(figsize=(6, 4))
sns.boxplot(x='HasBasement', y='SalePrice', data=train)
plt.title('Basement Presence vs Sale Price')
plt.savefig('charts/basement_vs_saleprice.png')
plt.close()

# Q14: How do remodeling/renovations influence value over time? 
train['RemodDecade'] = (train['YearRemodAdd'] // 10) * 10  # group remodel years into decades
print("\n--- Q14: Average SalePrice by remodel decade ---")
print(train.groupby('RemodDecade')['SalePrice'].mean())

plt.figure(figsize=(8, 5))
train.groupby('RemodDecade')['SalePrice'].mean().plot(kind='bar')
plt.ylabel('Average SalePrice')
plt.title('Average Sale Price by Remodel Decade')
plt.tight_layout()
plt.savefig('charts/remodel_decade_vs_saleprice.png')
plt.close()
# Insight: Houses remodeled more recently generally sell for higher prices,showing that renovation timing does add value.
 
#  Q15: Can we predict house prices using ONLY categorical features? 
cat_only_encoded = pd.get_dummies(train[categorical_cols], drop_first=True)
X = cat_only_encoded
y = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"\n--- Q15: R2 score using ONLY categorical features: {r2:.4f} ---")
# Insight: A low R2 score here would show that categorical features alone are not enough to predict price well - 
# numerical features like TotalBsmtSF and LotArea are needed for a good model.

print("\nAll charts saved in the 'charts' folder.")
print("Task 3 completed successfully.")
