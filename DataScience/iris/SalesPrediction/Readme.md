# Sales Prediction Using Python and Machine Learning

Predict product **Sales** from advertising spend on **TV**, **Radio**, and **Newspaper** using Linear Regression and Random Forest, and identify which channels appear most influential.

## Table of Contents

1. [Objective](#1-objective)
2. [Dataset](#2-dataset)
3. [Tools and Libraries](#3-tools-and-libraries)
4. [Project Files](#4-project-files)
5. [Setup and How to Run](#5-setup-and-how-to-run)
6. [Notebook Structure](#6-notebook-structure)
7. [Methodology](#7-methodology)
8. [Results](#8-results)
9. [Limitations](#9-limitations)
10. [Author](#10-author)

---

## 1. Objective
Build and compare regression models that predict **Sales** from advertising budgets, evaluate them on a held-out test set, analyse residuals, and interpret which advertising channels matter most, while noting that association does not prove causation.

## 2. Dataset
- **File:** `Advertising.csv` (classic Advertising dataset)
- **Sources:** Kaggle (for example `ashydv/advertising-dataset`) or the *An Introduction to Statistical Learning* (ISLR) website
- **Features:** `TV`, `Radio`, `Newspaper` (advertising spend)
- **Target:** `Sales`
- **Notes:**
  - The notebook checks the column names after loading and drops the `Unnamed: 0` index column if present.
  - No data is fabricated. If the file is missing, the notebook tries to download it from the ISLR site and stops with a clear message if that fails.
  - In the original dataset, spend is in thousands of dollars and Sales in thousands of units.

## 3. Tools and Libraries
Python 3.9+, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter.

## 4. Project Files

```
.
├── README.md
├── Sales_Prediction_Advertising.ipynb
└── Advertising.csv        # download separately
```

## 5. Setup and How to Run

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Launch Jupyter
jupyter notebook
```

1. Place `Advertising.csv` in the same folder as the notebook, or edit `DATA_PATH` in the dataset loading cell.
2. Open `Sales_Prediction_Advertising.ipynb`.
3. Select **Kernel → Restart & Run All**.

## 6. Notebook Structure
1. Project Title and Objective
2. Import Libraries
3. Dataset Loading
4. Data Cleaning and Inspection
5. Exploratory Data Analysis
6. Data Visualizations
7. Correlation and Feature Analysis
8. Train/Test Split
9. Linear Regression Model
10. Random Forest Regression Model
11. Evaluation Metrics and Comparison
12. Best Model Selection
13. Residual Analysis
14. Advertising Channel Impact
15. Sample Sales Prediction
16. Conclusion and Limitations

## 7. Methodology
- **Exploration:** first rows, shape, data types, missing values, duplicates, descriptive statistics.
- **Visualisation:** pairplot, Sales vs. each channel with trend lines, annotated correlation heatmap.
- **Split:** 80/20 using `train_test_split` with `random_state=42`, done before any learned preprocessing or modelling to avoid data leakage.
- **Models:**
  - Linear Regression (baseline)
  - Random Forest Regressor (alternative, `random_state=42`)
  - Both sit in pipelines with a median imputer fitted on training data only.
- **Evaluation:** MAE, RMSE, and R² on the same test set.
  - Lower MAE and RMSE and higher R² generally mean better performance.
- **Model selection:** chosen from the computed results (lowest RMSE, then MAE, then checked against R²), with no hardcoded assumptions.
- **Extras:** 5-fold cross-validation on training data only, standardised coefficients, and permutation importance.
- **Diagnostics:** actual vs. predicted plot, and a residual plot with a zero reference line.
- **Interpretation:** linear coefficients, Random Forest feature importances, and a sample prediction for a labelled hypothetical budget.

## 8. Results
Results are produced when the notebook runs on your dataset. The final summary cell prints the metrics, the selected model, and the most influential channel. Fill in your numbers after running:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | *fill in* | *fill in* | *fill in* |
| Random Forest | *fill in* | *fill in* | *fill in* |

- **Selected model:** *fill in*
- **Most influential channel:** *fill in*
- **Sample prediction (TV=150, Radio=25, Newspaper=30):** *fill in*

## 9. Limitations
- The dataset is small (about 200 rows), so test metrics can vary with the split.
- Results show association, not causation.
- Correlated channels and feature units affect how coefficients should be read.
- Price, seasonality, region, and competition are not in the data.
- Predictions outside the observed spend ranges are unreliable.
- The Random Forest was only lightly tuned.

## 10. Author
- **Name:** *your name*
- **Contact:** *email / LinkedIn / GitHub*