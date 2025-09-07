# Binary Classification Anomaly Detection using Logistic Regression and Random Forest

# 0. User variables
common_path = "./notebooks/DATABRICKS-PLAYGROUND"
input_file_path = common_path + "/data/input/warnings_vin_sample.csv"
output_file_path = common_path + "./data/output/result.csv"
logging_file_path = common_path + "./data/output/result.log"

# 1. Import Libraries

"""
Importing libraries is essential to prepare functions and tools
for data manipulation (pandas, numpy), visualization (matplotlib, seaborn),
statistical operations (statsmodels, scipy), machine learning (sklearn),
and evaluation metrics.

Accurate imports ensure reproducibility and usage of appropriate algorithms
and tests customized for anomaly detection contexts.
"""

## 1.0 Logging
import os                             # File and operating system utilities
import logging                        # Logging information

log_dir = os.path.dirname(logging_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # console output
        logging.FileHandler(logging_file_path, mode='w')  # write log to file, overwrite each run
    ]
)

## 1.1 Common Libraries
import pandas as pd                   # Data manipulation & analysis
import numpy as np                    # Numerical computations
import matplotlib.pyplot as plt       # Visualization
import seaborn as sns                 # Statistical data visualization
import math                           # Built-in mathematical functions

## 1.2 Statistical Methods
from statsmodels.stats.proportion import proportion_confint # Confidence intervals for proportions (e.g., Wilson’s CI)

## 1.3 Data Splitting
from sklearn.model_selection import train_test_split # Split data into training and testing sets

## 1.4 Preprocessing
from sklearn.preprocessing import StandardScaler              # Feature scaling
from sklearn.preprocessing import OneHotEncoder               # Categorical encoding
from sklearn.compose import ColumnTransformer                 # Apply transformations to columns
from sklearn.pipeline import Pipeline                         # Create preprocessing + modeling pipelines
from sklearn.impute import SimpleImputer                      # Handle missing data

## 1.5 Machine Learning Models
from sklearn.linear_model import LogisticRegression           # Supervised learning: Logistic Regression Classifier
from sklearn.ensemble import RandomForestClassifier           # Supervised learning: Random Forest Classifier

## 1.6 Evaluation Metrics for the ML Models
from sklearn.metrics import classification_report             # Detailed classification metrics
from sklearn.metrics import roc_auc_score                     # Area Under ROC curve
from sklearn.metrics import confusion_matrix                  # Compute confusion matrix
from sklearn.metrics import balanced_accuracy_score           # Balanced accuracy (for imbalanced data)
from sklearn.metrics import precision_score                   # Precision
from sklearn.metrics import average_precision_score           # Average precision (useful for PR curve)
from sklearn.metrics import recall_score                      # Recall
from sklearn.metrics import f1_score                          # F1-score

logging.info("Starting anomaly detection script")

# 2. Data Loading

"""
Load the data into memory and parse dates.

Correct loading is foundational as downstream steps rely on proper data types and formats,
especially dates which can be used for temporal anomaly trend analysis.

Column structure inspection helps confirm dataset consistency against expectations.
"""

path = input_file_path
logging.info("Loading data from %s", path)

try:
    df = pd.read_csv(path, sep=";")
except Exception as e:
    logging.error("Error loading data: %s", e)
    raise e

df["reporting_date"] = pd.to_datetime(df["reporting_date"], errors="coerce")
logging.info("Converted reporting_date to datetime")

# 3. Sanity Checks

"""
Sanity checks validate dataset integrity before modeling.

- Shape and info ensure dataset loaded fully and data types are appropriate.
- Summary statistics give insight into distributions, ranges, and missing data patterns.
- Identification of constant columns prevents redundant features causing noise.
- Splitting categorical and numeric helps in later tailored preprocessing.
- High-cardinality features indicate need for special encoding (frequency or target encoding).
- Duplicate row check guards against data leakage or bias.
- Target distribution review ensures meaningful class representation for supervised learning.
- Sanity checks for specific columns like VIN confirm domain-specific expectations.
"""

logging.info("Running basic sanity checks...")

## 3.1 Basic shape and info
logging.info(f"Dataframe shape: {df.shape}")
logging.info(f"Dataframe info: {df.info()}")

## 3.2 Summary statistics
'''
5-Number Statistics: 
One of the quickest methods for getting a feel for new data is the 5-number summary. 
It prints out 5 metrics about a distribution - the minimum, 25th percentile, median, 75th percentile, and the maximum along with mean and standard deviation. 
By looking at the 5-number summary and the difference between the mean and the minimum/maximum values, you can get a rough idea of whether outliers are present in the distribution.
'''

# Set the target variable
target_variable = "anomaly_flag"
logging.info(f"5 Number statistics of {target_variable} is: {df[target_variable].describe()}")

## 3.3 Constant columns (all same value)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
logging.info(f"Constant columns: {constant_cols}")

## 3.4 High Cardinatliy Categoricals
num_cols = df.select_dtypes(include=["number"]).columns.tolist() # Numerical type columns
cat_cols = [c for c in df.columns if c not in num_cols] # Category type columns
high_card = [c for c in cat_cols if df[c].nunique() > 100]
logging.info(f"High-cardinality categoricals: {high_card}")

## 3.5 Feature info summary
feature_info = pd.DataFrame({
    "dtype": df.dtypes,
    "n_missing": df.isna().sum(),
    "% missing": (df.isna().sum() / len(df) * 100).round(2),
    "n_unique": df.nunique(),
})
logging.info(f"Feature Information: {feature_info}")

## 3.6 Duplicates at dataset level
n_dupes = df.duplicated().sum()
logging.info(f"Number of duplicate rows: {n_dupes}")

## 3.7 VIN sanity check
df_vin_contradict = df[df["ctt_vin"].isna() & (df["anomaly_description"].str.contains("No anomaly detected", na=False))]
logging.info(f"Records with missing VIN but 'No anomaly detected': {len(df_vin_contradict)}")

## 3.8 Data Visualizations

target_variable = "anomaly_flag"

### 3.8.1 Countplot for Binary Classification Outlier detection
plt.figure(figsize=(6, 4))
sns.countplot(x=target_variable, data=df)
plt.title('Anomaly Flag Class Distribution')
plt.xlabel('Anomaly Flag')
plt.ylabel('Count')
plt.show()


# 4. Class Distribution Analysis

"""
Analysis of class balance is critical in anomaly detection characterized by rare positive samples.

- Class imbalance affects model training and evaluation.
- Prevalence and confidence intervals are statistical measures helping assess data sufficiency and reliability.
- Wilson's method for confidence intervals is robust for binomial proportions often used for anomaly labels.
- Standard error quantifies uncertainty aiding considered thresholding and risk assessment.
"""

logging.info("Analyzing class imbalance")

## 4.1 Class imbalance
class_counts = df["anomaly_flag"].value_counts()
logging.info(f"Class counts: {class_counts.to_dict()}")

## 4.2 Class Imbalance Ratio (CIR)
target = "anomaly_flag"
n0, n1 = df[target].value_counts().values
cir = max(n0, n1) / min(n0, n1)
logging.info(f"Class imbalance ratio: {cir}")

## 4.3 Prevalence
n_total = len(df)
n_pos = n1
prevalence = n_pos / n_total
logging.info(f"Prevalence: {prevalence}")

## 4.4 Wilsons 95% CI
ci_low, ci_high = proportion_confint(count=n_pos, nobs=n_total, method="wilson")
logging.info(f"Wilson 95%% CI for prevalence: low: {ci_low}, high: {ci_high}")

## 4.5 Standard Error
SE = (prevalence * (1 - prevalence) / n_total) ** 0.5
logging.info(f"Standard Error: {SE}")

# 5. Data Sufficiency

"""
Calculate sample size and required positive/negative cases to achieve a margin of error at specified confidence levels.

Ensures dataset size and class proportions provide statistically significant model training
and avoid misleading results from insufficient data.
"""

logging.info("Calculating required sample size for proportion estimates")

## 5.1 Function for calculating prop
def required_n_for_proportion(m=0.05, confidence=0.95, p=None):
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    # Worst-case prevalence if not provided
    if p is None:
        p = 0.5
    
    n_needed = math.ceil((z**2) * p * (1 - p) / (m**2))
    return n_needed

## 5.2 Apply to your dataset: 95% CI, ±5% margin of error
n_needed = required_n_for_proportion(m=0.05, confidence=0.95)
p_hat = df["anomaly_flag"].mean()
n_pos_needed = math.ceil(n_needed * p_hat)
n_neg_needed = n_needed - n_pos_needed

logging.info(f"Required sample size total: {n_needed}")
logging.info(f"Required positives: {n_pos_needed}")
logging.info(f"Required negatives: {n_neg_needed}")

# 6. Feature Selection

"""
Select relevant features while preventing data leakage and redundancy.

- Drop constant, leakage-prone columns (e.g. anomaly_description heavily correlated with label).
- Preserve domain-important identifiers (ctt_vin, ctt_cms_contract_number).
- Remove highly correlated numeric features to reduce multicollinearity.
- This process improves model generalization and interpretability.
"""

logging.info("Calculating correlation with target")

## 6.1 Correlation with target (numeric only)
cor_target = df.corr(numeric_only=True)["anomaly_flag"].abs().sort_values(ascending=False)
logging.info(f"Correlation with target variable '{target_variable}' is: {cor_target}")

## 6.2 Candidate columns to drop

### Identify columns to drop based on criteria
drop_cols = []

cols = df.columns

### Drop columns with single unique value (constant)
for col in cols:
    if df[col].nunique() == 1:
        drop_cols.append(col)

### Drop columns likely causing data leakage or not useful
leakage_candidates = ['anomaly_description']  # Text describing target, high leakage risk
drop_cols += [col for col in leakage_candidates if col in cols]

### Final drop columns ensuring to keep important ID columns
ids_to_keep = ['ctt_cms_contract_number', 'ctt_vin']
drop_cols = [col for col in drop_cols if col not in ids_to_keep]

logging.info(f"Dropping columns: {drop_cols}")

## 6.3 Remove identified columns
df_model = df.drop(columns=list(drop_cols))

## 6.4 Remove highly correlated numeric features (multicollinearity check)
### Create correlation matrix for features only (exclude target)
### Select only numeric columns for correlation matrix
numeric_feats = df_model.select_dtypes(include=[np.number]).columns.tolist()

### Exclude target column if present
if "anomaly_flag" in numeric_feats:
    numeric_feats.remove("anomaly_flag")

### Now compute correlation matrix only on numeric features
corr_matrix = df_model[numeric_feats].corr().abs()

### Upper triangle mask to avoid duplicated pairs and diagonal
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

### Identify highly correlated columns
high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
logging.info(f"Dropping highly correlated columns: {high_corr}")

## 6.5 Final
df_model = df_model.drop(columns=high_corr)
logging.info(f"Final dataframe shape for modeling: {df_model.shape}")

# 7. Train-test Split

"""
Create training and testing subsets sustaining class balance (stratification).

This step mimics real-world data division, allowing model validation on unseen data,
particularly preserving rare anomaly instances proportionally in both subsets.
"""

logging.info("Performing train-test split with stratification")

## 7.1 Check balance in y
X = df_model.drop(columns=["anomaly_flag"]) # Keep as DataFrame
y = df_model["anomaly_flag"] # Keep as series
logging.info(f"Proportion of anomalies: {y.mean()}")

## 7.2 Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

logging.info(f"Train set size: {X_train.shape}; Test set size: {X_test.shape}")
logging.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
logging.info(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

## 7.3 Data Visualizations
### 7.3.1 Class distribution in train and test sets side by side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train)
plt.title('Train Set Class Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x=y_test)
plt.title('Test Set Class Distribution')
plt.show()

# 8. Preprocessing

"""
Prepare data pipelines for numeric/categorical features:

- Numeric features imputed (median) and scaled (StandardScaler) for uniformity improving model convergence.
- Categorical features imputed (mode) and one-hot encoded for algorithm compatibility without numerical assumptions.
- Combined preprocessing via ColumnTransformer ensures structured handling within sklearn pipeline.
"""

logging.info("Setting up preprocessing pipelines")

## 8.1 Identify column types
numeric_features = X_train.select_dtypes(include=["int64", "float64", "number"]).columns
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns

logging.info(f"Numeric features: {list(numeric_features)}")
logging.info(f"Categorical features: {list(categorical_features)}")

## 8.2 Setting up pipleines

### 8.2.1 Numeric type

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

### 8.2.2 Categorical type
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # keep sparse default
])

### 8.2.3 Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 9. Machine Learning Models

"""
Train classifiers Logistic Regression and Random Forest:

- Logistic regression is robust, interpretable linear model suitable for classification.
- Random forest captures nonlinear feature interactions and generally achieves higher accuracy.
- Class imbalance addressed by class_weight='balanced' option, recalibrating model to minority anomaly instances.
- Train/test score printing assists performance inspection and potential overfitting detection.
"""

## 9.1 Round 1

### 9.1.1 Logistic Regression

logging.info("Training Logistic Regression model (round 1)")

log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
])

log_reg.fit(X_train, y_train)

logging.info(f"Train accuracy: {log_reg.score(X_train, y_train)}")
logging.info(f"Test accuracy: {log_reg.score(X_test, y_test)}")

### 9.1.2 Random Forest

logging.info("Training Random Forest model (round 1)")

rf = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

rf.fit(X_train, y_train)

logging.info(f"Train accuracy: {rf.score(X_train, y_train)}")
logging.info(f"Test accuracy: {rf.score(X_test, y_test)}")

## 9.2 Round 2: Addressing Class Imbalance

### 9.2.1 Logistic Regression

logging.info("Training Logistic Regression model (class imbalance addressed)")

log_reg_2 = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
]).fit(X_train, y_train)

log_reg_2.fit(X_train, y_train)

logging.info(f"Train accuracy: {log_reg_2.score(X_train, y_train)}")
logging.info(f"Test accuracy: {log_reg_2.score(X_test, y_test)}")

### 9.2.2 Random Forest

logging.info("Training Random Forest model (class imbalance addressed)")

rf_2 = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
]).fit(X_train, y_train)

rf_2.fit(X_train, y_train)

logging.info(f"Train accuracy: {rf_2.score(X_train, y_train)}")
logging.info(f"Test accuracy: {rf_2.score(X_test, y_test)}")

# 10. Model Comparison

"""
Evaluate models with multiple metrics:

- ROC AUC quantifies model discrimination capability.
- Precision, recall, F1 reflect correctness and completeness of anomaly identification.
- Average precision exemplifies performance under class imbalance (like PR AUC).
- Balanced accuracy balances sensitivity and specificity.
- Classification report and confusion matrix detail model outputs per class.
"""

logging.info("Evaluating model performance")

## 10.1 Metrics: ROC, AUC, Precision, Recall, Average Precision, F1-Score, Balanced Accuracy
def evaluate(model, X_train, X_test, y_train, y_test):

    # training results
    y_pred_train = model.predict(X_test)
    y_prob_train = model.predict_proba(X_test)[:,1]

    # test results 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # train accuracy
    train_acc = model.score(X_train, y_train)

    # test accuracy
    test_acc = model.score(X_test, y_test)

    metrics = {
        "train_accuray": train_acc,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }
    
    logging.info(f"Evaluation metrics: {metrics}")
    logging.info(f"Classification report: {classification_report(y_test, y_pred)}")
    logging.info(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")
    
    return metrics

## 10.2 Full Model Evaluation

results = []

### 10.2.1 Logistic Regression
metrics_lr = evaluate(log_reg, X_train, X_test, y_train, y_test)
metrics_lr.update(
    {"model": "Logistic Regression", "stage": "Before Balancing Classes"}
)
results.append(metrics_lr)

metrics_lr_2 = evaluate(log_reg_2, X_train, X_test, y_train, y_test)
metrics_lr_2.update(
    {"model": "Logistic Regression", "stage": "After Balancing Classes"}
)
results.append(metrics_lr_2)

### 10.2.2 Random Forest
metrics_rf = evaluate(rf, X_train, X_test, y_train, y_test)
metrics_rf.update(
    {"model": "Random Forest", "stage": "Before Balancing Classes"}
)
results.append(metrics_rf)

metrics_rf_2 = evaluate(rf_2, X_train, X_test, y_train, y_test)
metrics_rf_2.update(
    {"model": "Random Forest", "stage": "After Balancing Classes"}
)
results.append(metrics_rf_2)

results_df = pd.DataFrame(results)

results_df.to_csv(output_file_path, index=False)

## 10.3 Data Visualizations

### 10.3.1 ROC Curves for both models
from sklearn.metrics import roc_curve

plt.figure(figsize=(8,6))

for name, model in [('Logistic Regression', log_reg), ('Random Forest', rf)]:
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})')

plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()