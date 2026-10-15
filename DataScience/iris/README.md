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

# 📧 Email Spam Detection using Machine Learning & NLP

A beginner-friendly machine learning project that classifies messages as **spam** or **ham** (legitimate) using **TF-IDF** features and two classifiers: **Multinomial Naive Bayes** and **Logistic Regression**.

---

## 📌 Project Overview

Spam messages waste time and can carry scams or phishing links. This project builds a binary text classifier that learns to separate spam from legitimate messages, evaluates it on a held-out test set, and uses it to predict new messages.

**Goals**
- Explore and understand a real-world text dataset
- Clean and preprocess text without losing useful spam signals
- Convert text into numeric features with TF-IDF
- Train and compare two classification models
- Evaluate with precision, recall, F1-score and confusion matrices, not accuracy alone

---

## 📊 Dataset

**SMS Spam Collection** dataset, labelled as `ham` or `spam`.

| Source | Link |
|---|---|
| UCI Machine Learning Repository | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Kaggle | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset |

- **UCI file:** `SMSSpamCollection` (tab-separated, no header)
- **Kaggle file:** `spam.csv` (comma-separated, latin-1 encoding)

The notebook detects the label and message columns automatically, so either format works. Place the file next to the notebook (or in a `data/` folder), or set `DATA_PATH` in the notebook. If no file is found, it tries to download the UCI version.

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Data:** pandas, NumPy
- **ML:** scikit-learn
- **Visualization:** Matplotlib, Seaborn, WordCloud (optional)
- **NLP:** NLTK stopwords (optional, falls back to scikit-learn's list)

---

## 🔄 Project Workflow

1. **Data loading:** detect file format, label column and message column
2. **Exploration:** shape, missing values, duplicates, class distribution
3. **Cleaning:** remove conflicting and duplicate messages
4. **Text preprocessing:** lowercase, punctuation removal, stopword removal (digits and a currency marker are kept as spam signals)
5. **Feature extraction:** TF-IDF (unigrams + bigrams) inside a `Pipeline`, so there is no data leakage
6. **Train/test split:** 80/20, `random_state=42`, stratified
7. **Model training:** Multinomial Naive Bayes and Logistic Regression
8. **Evaluation:** accuracy, precision, recall, F1, classification reports, confusion matrices
9. **Model selection:** best model chosen by spam F1-score (recall as tie-breaker)
10. **Prediction:** labels and spam probabilities for new messages

---

## 🤖 Models Used

| Model | Why it's used |
|---|---|
| **Multinomial Naive Bayes** | Fast, simple baseline that suits text features |
| **Logistic Regression** | Linear model with interpretable word weights and reliable probabilities |

---

## 📈 Results

Fill this table in after running the notebook (Section 9, *Model Comparison*):

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Multinomial Naive Bayes | – | – | – | – |
| Logistic Regression | – | – | – | – |

**Best model:** _fill in from the notebook output_

> Spam is the minority class, so recall and precision for spam matter more than accuracy. Low recall lets spam into the inbox, and low precision hides legitimate messages.

---

## 📁 Project Structure

```
email-spam-detection/
│
├── Email_Spam_Detection.ipynb   # Main notebook
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── .gitignore
└── data/                        # Dataset (not tracked by Git)
    └── SMSSpamCollection
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/email-spam-detection.git
cd email-spam-detection
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**
Download it from UCI or Kaggle (links above) and place it in the project folder or `data/`.

**5. Run the notebook**
```bash
jupyter notebook Email_Spam_Detection.ipynb
```
Then choose **Kernel → Restart & Run All**.

---

## 🔮 Example Prediction

```python
new_messages = ["Congratulations! You won a free prize. Call now to claim!"]
cleaned = [clean_text(m) for m in new_messages]
best_model.predict(cleaned)          # 1 = spam, 0 = ham
best_model.predict_proba(cleaned)    # [P(ham), P(spam)]
```

---

## ⚠️ Limitations

- Small, older dataset; modern spam and phishing may look different
- Results come from a single train/test split, so small differences between models may be noise
- TF-IDF ignores meaning and word order beyond two-word phrases
- Hyper-parameters use sensible defaults and are not exhaustively tuned

---

## 🚀 Future Improvements

- Hyper-parameter tuning with `GridSearchCV`
- More models (linear SVM, Random Forest)
- Extra features (message length, capital-letter ratio, URL/phone detection)
- Word embeddings or transformer models (e.g. BERT)
- Deploy as a web app (Streamlit or Flask)

---

## 🙌 Acknowledgements

- Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. — *SMS Spam Collection* dataset (UCI Machine Learning Repository)
- scikit-learn, pandas, Matplotlib, Seaborn, NLTK and WordCloud libraries

---

⭐ If you found this project helpful, consider giving it a star!

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/PranaliVeer935)
- LinkedIn: [your-profile](https://www.linkedin.com/in/pranali-veer-a54ab7353)

  # Sales Prediction Using Python and Machine Learning

Predict product **Sales** from advertising spend on **TV**, **Radio**, and **Newspaper** using Linear Regression and Random Forest, and identify which channels appear most influential.

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
├── .gitignore
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
