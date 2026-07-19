# =====================================================
# TASK 4: MODEL BUILDING
# =====================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

pd.set_option('display.max_columns', None)
os.makedirs('charts', exist_ok=True)

# Step 1: Load data and split into train/test
df = pd.read_excel('HousePrediction.xlsx')
train = df[df['SalePrice'].notnull()].copy()
test = df[df['SalePrice'].isnull()].copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Step 2: Handle missing values in the test set
test['MSZoning'] = test['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(train['TotalBsmtSF'].median())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(train['BsmtFinSF2'].median())

print("\nMissing values in test after filling:")
print(test.isnull().sum())

# Step 3: Feature engineering
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis=0)

categorical_cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

reference_year = train['YearRemodAdd'].max()
combined_encoded['HouseAge'] = reference_year - combined_encoded['YearBuilt']
combined_encoded['YearsSinceRemod'] = reference_year - combined_encoded['YearRemodAdd']

train_final = combined_encoded[combined_encoded['is_train'] == 1].drop(columns=['is_train'])
test_final = combined_encoded[combined_encoded['is_train'] == 0].drop(columns=['is_train'])

print("\nFinal train shape after encoding:", train_final.shape)
print("Final test shape after encoding:", test_final.shape)

# Step 4: Prepare X (features) and y (target)
drop_cols = ['Id', 'SalePrice']
X = train_final.drop(columns=drop_cols)
y = train_final['SalePrice']

X_test_final = test_final.drop(columns=drop_cols)  # SalePrice is all NaN here anyway

# Step 5: Split train data into train/validation to check model performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train and compare two simple models

#  Model 1: Linear Regression 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_val)

lr_r2 = r2_score(y_val, lr_pred)
lr_mae = mean_absolute_error(y_val, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))

print("\n--- Linear Regression Performance (on validation set) ---")
print(f"R2 Score: {lr_r2:.4f}")
print(f"MAE: {lr_mae:.2f}")
print(f"RMSE: {lr_rmse:.2f}")

# Model 2: Random Forest Regressor 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)

rf_r2 = r2_score(y_val, rf_pred)
rf_mae = mean_absolute_error(y_val, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))

print("\n--- Random Forest Performance (on validation set) ---")
print(f"R2 Score: {rf_r2:.4f}")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")

# Step 7: better model based on R2 score
if rf_r2 > lr_r2:
    print("\nRandom Forest performed better - using it as the final model.")
    final_model = rf_model
    best_model_name = 'Random Forest'
else:
    print("\nLinear Regression performed better - using it as the final model.")
    final_model = lr_model
    best_model_name = 'Linear Regression'

final_model.fit(X, y)

# Step 8: Feature importance 
if best_model_name == 'Random Forest':
    importance = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top 10 Most Important Features ---")
    print(importance.head(10))

    plt.figure(figsize=(8, 6))
    importance.head(10).sort_values().plot(kind='barh')
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('charts/feature_importance.png')
    plt.close()

# Step 9: Predict SalePrice for the test set and save results
test_predictions = final_model.predict(X_test_final)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")
print(submission.head())

val_pred_final = rf_pred if best_model_name == 'Random Forest' else lr_pred
plt.figure(figsize=(7, 7))
plt.scatter(y_val, val_pred_final, alpha=0.4)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # perfect prediction line
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title(f'Actual vs Predicted SalePrice ({best_model_name})')
plt.tight_layout()
plt.savefig('charts/actual_vs_predicted.png')
plt.close()

print(f"\nFinal model used: {best_model_name}")
print("Task 4 completed successfully.")
