# 🏠 House Price Prediction — Data Analysis & Regression Modeling

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Status](https://img.shields.io/badge/Status-Completed-1D9E75?style=for-the-badge)

An end-to-end data analysis and machine learning project that explores the Ames Housing dataset, cleans and engineers its features, answers 15 structured business questions, and builds a regression model to predict house sale prices — all in Python.

---

## 🖥️ Project Overview

This project walks through a complete analytics workflow on a real estate dataset:

| Stage | Focus Area |
|---|---|
| **Task 1** | Basic data exploration — structure, columns, data types |
| **Task 2** | Data cleaning — missing values, temporal features, distributions, log transformation |
| **Task 3** | Feature engineering — outliers, categorical/numerical relationships, correlation, encoding, scaling |
| **Task 4** | Model building — Linear Regression vs Random Forest, feature importance, final predictions |

---

## 📁 Project Structure

```
house-price-prediction/
│
├── 📂 data/
│   └── HousePrediction.xlsx                    # Raw dataset (2,919 rows)
│
├── 📂 scripts/
│   ├── task1_basic_exploration.py               # Data loading & column insights
│   ├── task2_data_cleaning.py                   # Missing values, distributions, log transform
│   ├── task3_feature_engineering.py              # Outliers, encoding, correlation, Q11–Q15
│   └── task4_model_building.py                   # Model training, comparison & predictions
│
├── 📂 charts/
│   └── *.png                                     # All generated visualizations
│
├── 📂 output/
│   └── predictions.csv                           # Final SalePrice predictions (test set)
│
├── 📂 report/
│   └── House_Price_Prediction_Report.docx        # Full project documentation
│
└── README.md
```

---

## 📌 Problem Statement

Estimating a fair house price is a common challenge for sellers, buyers, and real estate agents, who often rely on gut feeling rather than data. This project uses historical property records to:

- Identify which physical, structural, and location-based features actually drive sale price
- Quantify relationships that are often assumed but rarely tested (e.g., does condition rating matter as much as people think?)
- Build a regression model capable of predicting price for unseen properties

---

## 🗂️ Dataset Overview

| Property | Detail |
|---|---|
| **Source** | Excel Workbook (`.xlsx`) — Ames Housing dataset (reduced feature set) |
| **Total Rows** | 2,919 |
| **Labeled Rows (train)** | 1,460 — used for all analysis and modeling |
| **Unlabeled Rows (test)** | 1,459 — held out, used only for final predictions |
| **Columns** | 13 |
| **Target Variable** | `SalePrice` (continuous) |

### Key Fields
`MSSubClass` · `MSZoning` · `LotArea` · `LotConfig` · `BldgType` · `OverallCond` · `YearBuilt` · `YearRemodAdd` · `Exterior1st` · `BsmtFinSF2` · `TotalBsmtSF` · `SalePrice`

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3** | Core language |
| **pandas / numpy** | Data manipulation and cleaning |
| **matplotlib / seaborn** | Visualization and EDA |
| **scikit-learn** | Model training, evaluation, preprocessing |
| **python-docx (via docx)** | Automated report generation |

---

## 🔧 Data Cleaning & Feature Engineering

- Split the combined dataset into **train** (has `SalePrice`) and **test** (missing `SalePrice`) subsets
- Checked missing values across the full dataset — only 6 missing values total, all in the test subset
- Filled test-set missing values using **train-derived statistics only** (mode for categorical, median for numerical) to avoid data leakage
- Applied **log1p transformation** to right-skewed `SalePrice` and `LotArea`
- Detected outliers using the **IQR method** across all continuous features
- **One-hot encoded** categorical columns (`MSZoning`, `LotConfig`, `BldgType`, `Exterior1st`)
- **Scaled** continuous numerical features using `StandardScaler`
- Engineered temporal features: `HouseAge` and `YearsSinceRemod` from `YearBuilt` / `YearRemodAdd`

---

## 📊 Correlation & Key Relationships

| Feature | Correlation with SalePrice |
|---|---|
| `TotalBsmtSF` | **0.61** |
| `YearBuilt` | **0.52** |
| `YearRemodAdd` | 0.51 |
| `LotArea` | 0.26 |
| `OverallCond` | -0.08 |
| `BsmtFinSF2` | -0.01 |

---

## 💡 Key Insights

| # | Insight |
|---|---|
| 1 | `TotalBsmtSF` and `YearBuilt` are the strongest numerical drivers of sale price |
| 2 | `OverallCond`, despite being an intuitive quality measure, is a weak standalone predictor (correlation ≈ -0.08) |
| 3 | Homes **with** a basement average **₹182,878** vs **₹105,653** without one — a ~73% price gap |
| 4 | `MSZoning` creates large price gaps — **FV (Floating Village Residential)** commands the highest premium |
| 5 | Location alone isn't enough — **FV zoning + Cul-de-Sac lot configuration** together produce the highest average price |
| 6 | Average sale price rises consistently by remodel decade — from ~₹121K (1950s) to ~₹337K (2010s) |
| 7 | Categorical features alone explain only **25%** of price variance (R² = 0.25) — numerical features are essential |
| 8 | `SalePrice` and `LotArea` are right-skewed and require log transformation before use in linear models |

---

## 🤖 Model Building & Results

Two regression models were trained on an 80/20 train-validation split and compared:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.62 | ₹34,123 | ₹54,023 |
| **Random Forest Regressor** | **0.82** | **₹22,906** | **₹37,269** |

**Random Forest** was selected as the final model — it captured non-linear feature interactions (like `OverallCond`'s inconsistent relationship with price) that Linear Regression could not. The model was retrained on the full labeled dataset and used to generate predictions for all 1,459 unseen test rows.

**Top features by importance:** `TotalBsmtSF` → `HouseAge` → `YearBuilt` → `LotArea`

---

## 🚀 How to Use

1. **Clone the repository**
   ```
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
   ```

3. **Run the scripts in order**
   ```
   python scripts/task1_basic_exploration.py
   python scripts/task2_data_cleaning.py
   python scripts/task3_feature_engineering.py
   python scripts/task4_model_building.py
   ```

4. **Check the outputs**
   - Charts are saved to `charts/`
   - Final predictions are saved to `output/predictions.csv`
   - Full write-up is in `report/House_Price_Prediction_Report.docx`

---

## 📋 Task Coverage

| Task Group | Sub-Questions | Status |
|---|---|---|
| Task 1 — Basic Exploration | 4 steps | ✅ Complete |
| Task 2 — Data Cleaning & Analysis | Q1–Q10 | ✅ Complete |
| Task 3 — Feature Engineering & Advanced Analysis | Q1–Q5, Q11–Q15 | ✅ Complete |
| Task 4 — Model Building | Model comparison + predictions | ✅ Complete |
| Documentation Report | Full Word document | ✅ Complete |

---

## 👩‍💻 Author

**Vaibhavi Hambire**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/vaibhavi-hambire)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:hambirevaibhavi21@gmail.com)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

⭐ **If you found this project useful, consider giving it a star!**
