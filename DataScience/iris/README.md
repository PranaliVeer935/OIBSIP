# 🌸 Iris Flower Classification Using Machine Learning

A beginner-friendly machine learning project that classifies iris flowers into one of three species (**Setosa**, **Versicolor**, **Virginica**) using four physical measurements. Built as part of a Data Science internship assignment.

## 📌 Objective

Train and evaluate classification models that predict the species of an iris flower from:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## 📊 Dataset

- **Source:** Loaded directly from `sklearn.datasets.load_iris()`, so no external download is needed.
- **Size:** 150 samples, 4 numeric features, 3 classes.
- **Balance:** 50 samples per species (perfectly balanced).
- **Quality:** No missing values; 1 duplicate row (kept in the analysis).

## 🛠️ Tech Stack

- Python 3
- Jupyter Notebook
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

## 🔍 Project Workflow

1. Load the dataset into a pandas DataFrame
2. Initial inspection and exploratory data analysis (EDA)
3. Visualizations: pair plot, box plots, correlation heatmap
4. Feature analysis, including feature importance from tree-based models
5. Stratified 80/20 train/test split (`random_state=42`)
6. Train three classifiers with scaling handled inside `Pipeline`s
7. Evaluate with accuracy, precision, recall, F1-score, confusion matrices and classification reports
8. Compare models and select the best one programmatically
9. Predict the species of a sample flower

## 🤖 Models Trained

| Model | Preprocessing |
|---|---|
| Logistic Regression | StandardScaler (in Pipeline), `max_iter=1000` |
| K-Nearest Neighbors | StandardScaler (in Pipeline), `n_neighbors=5` |
| Decision Tree | None needed, fixed `random_state` |

Scaling is fitted on the training data only, and the test set is used solely for evaluation, so there is no data leakage.

## 📈 Results

On the held-out test set (30 samples, `random_state=42`):

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 93.3% |
| Decision Tree | 93.3% |
| K-Nearest Neighbors | 93.3% |

**Key findings**

- Setosa is clearly separable from the other two species.
- All misclassifications occurred between Versicolor and Virginica.
- Petal length and petal width are the most discriminative features in this dataset; sepal width is the weakest on its own.
- The three models perform essentially the same on this small test set, so no model can be called meaningfully better based on this evidence alone.

> Results may vary slightly with different library versions. The values above come from the notebook with `random_state=42`.

## ⚠️ Limitations

- The Iris dataset is small and unusually well structured, so results do not automatically generalize to other flower datasets or real-world problems.
- The test set has only 30 samples, so one or two errors noticeably change the metrics.
- A single train/test split can be lucky or unlucky; cross-validation would give a more reliable estimate.

## 🚀 How to Run

1. **Clone or download** this repository.

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook Iris_Flower_Classification.ipynb
   ```
   You can also open it in VS Code or Google Colab.

4. Run all cells from top to bottom.

## 📁 Repository Structure

```
├── Iris_Flower_Classification.ipynb   # Main notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── LICENSE                            # License
└── .gitignore                         # Files ignored by Git
```


