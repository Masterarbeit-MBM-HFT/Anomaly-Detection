# Binary Classification Anomaly Detection using Logistic Regression and Random Forest

# 1. Import Libraries

"""
Importing libraries is essential to prepare functions and tools
for data manipulation (pandas, numpy), visualization (matplotlib, seaborn),
statistical operations (statsmodels, scipy), machine learning (sklearn),
and evaluation metrics.

Accurate imports ensure reproducibility and usage of appropriate algorithms
and tests customized for anomaly detection contexts.
"""

## 1.1 Common Libraries
import pandas as pd                   # Data manipulation & analysis
import numpy as np                    # Numerical computations
import os                             # File and operating system utilities
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

# 2. Data Loading

"""
Load the data into memory and parse dates.

Correct loading is foundational as downstream steps rely on proper data types and formats,
especially dates which can be used for temporal anomaly trend analysis.

Column structure inspection helps confirm dataset consistency against expectations.
"""

path = "./data/warnings_vin_sample.csv"
df = pd.read_csv(path, sep=";")
df["reporting_date"] = pd.to_datetime(df["reporting_date"], errors="coerce")
print(df.columns)

### Data Viz
#### Visualize feature missingness
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

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

## 3.1 Basic shape and info
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

## 3.2 Summary statistics
'''
5-Number Statistics: 
One of the quickest methods for getting a feel for new data is the 5-number summary. 
It prints out 5 metrics about a distribution - the minimum, 25th percentile, median, 75th percentile, and the maximum along with mean and standard deviation. 
By looking at the 5-number summary and the difference between the mean and the minimum/maximum values, you can get a rough idea of whether outliers are present in the distribution.
'''

# Set the target variable
target_variable = "anomaly_flag"
df[target_variable].describe()

## 3.3 Constant columns (all same value)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("\nConstant columns:", constant_cols)

## 3.4 Numeric and categorical split
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

### High-cardinality categoricals
high_card = [c for c in cat_cols if df[c].nunique() > 100]
print("\nHigh-cardinality categoricals:", high_card)

## 3.5 Feature info summary
feature_info = pd.DataFrame({
    "dtype": df.dtypes,
    "n_missing": df.isna().sum(),
    "% missing": (df.isna().sum() / len(df) * 100).round(2),
    "n_unique": df.nunique(),
})
print("\nFeature Information:")
print(feature_info)

## 3.6 Duplicates at dataset level
n_dupes = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {n_dupes}")

## 3.7 Target distribution (detailed anomaly_description)
print("\nAnomaly description distribution:")
for i, (k, v) in enumerate(df["anomaly_description"].value_counts().items()):
    print(f"{i}. {k}: {v}")

## 3.8 VIN sanity check
df_vin_contradict = df[
    df["ctt_vin"].isna() & 
    (df["anomaly_description"].str.contains("No anomaly detected", na=False))
]
print(f"\nRecords with missing VIN but 'No anomaly detected': {len(df_vin_contradict)}")

## 3.9 Data Visualizations

### 3.9.1 Countplot for Binary Classification Outlier detection

target_variable = "anomaly_flag"

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

## 4.1 Class imbalance
class_counts = df["anomaly_flag"].value_counts()
n0, n1 = class_counts[0], class_counts[1]

print("Majority class (0):", n0)
print("Minority class (1):", n1)
print("Class imbalance present? ->", n0 != n1)

## 4.2 Class Imbalance Ratio (CIR)
target = "anomaly_flag"
n0, n1 = df[target].value_counts().values
cir = max(n0, n1) / min(n0, n1)
print(f"Class imbalance ratio: {cir:.2f}:1")

## 4.3 Prevalence
n_total = len(df)
n_pos = n1
prevalence = n_pos / n_total
print(f"Prevalence of anomalies: {prevalence:.5f} ({100*prevalence:.3f}%)")

## 4.4 Wilsons 95% CI
ci_low, ci_high = proportion_confint(count=n_pos, nobs=n_total, method="wilson")
print(f"Wilson 95% CI for prevalence: [{ci_low:.5f}, {ci_high:.5f}]")

## 4.5 Standard Error
SE = (prevalence * (1 - prevalence) / n_total) ** 0.5
print(f"Standard Error of prevalence: {SE:.6f}")

# 5. Data Sufficiency

"""
Calculate sample size and required positive/negative cases to achieve a margin of error at specified confidence levels.

Ensures dataset size and class proportions provide statistically significant model training
and avoid misleading results from insufficient data.
"""

def required_n_for_proportion(m=0.05, confidence=0.95, p=None):
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    # Worst-case prevalence if not provided
    if p is None:
        p = 0.5
    
    n_needed = math.ceil((z**2) * p * (1 - p) / (m**2))
    return n_needed

## Apply to your dataset
## 95% CI, ±5% margin of error
n_needed = required_n_for_proportion(m=0.05, confidence=0.95)

## Expected prevalence from data
p_hat = df["anomaly_flag"].mean()

## Adjusted required positives (if prevalence is low)
n_pos_needed = math.ceil(n_needed * p_hat)
n_neg_needed = n_needed - n_pos_needed

print(f"Required total sample size: {n_needed}")
print(f"Given prevalence={p_hat:.5f}:")
print(f"Required positives: {n_pos_needed}")
print(f"Required negatives: {n_neg_needed}")

# 6. Feature Selection

"""
Select relevant features while preventing data leakage and redundancy.

- Drop constant, leakage-prone columns (e.g. anomaly_description heavily correlated with label).
- Preserve domain-important identifiers (ctt_vin, ctt_cms_contract_number).
- Remove highly correlated numeric features to reduce multicollinearity.
- This process improves model generalization and interpretability.
"""

## 6.1 Correlation with target (numeric only)
cor_target = df.corr(numeric_only=True)["anomaly_flag"].abs().sort_values(ascending=False)
print("Correlation with target:\n", cor_target)

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

print(f"Columns to drop based on logic: {drop_cols}")

## 6.3 Remove identified columns
df_model = df.drop(columns=list(drop_cols))
print(df_model.columns)

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
print("Highly correlated numeric features (r > 0.95) to drop:", high_corr)

## 6.5 Final
df_model = df_model.drop(columns=high_corr)
print("\nFinal modeling DataFrame shape:", df_model.shape)

# 7. Train-test Split

"""
Create training and testing subsets sustaining class balance (stratification).

This step mimics real-world data division, allowing model validation on unseen data,
particularly preserving rare anomaly instances proportionally in both subsets.
"""

## 7.1 Check balance in y
X = df_model.drop(columns=["anomaly_flag"]) # Keep as DataFrame
y = df_model["anomaly_flag"] # Keep as series
print("Proportion of anomalies:", y.mean())

## 7.2 Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Anomalies in train:", y_train.sum())
print("Anomalies in test:", y_test.sum())

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

## Identify column types
numeric_features = X_train.select_dtypes(include=["int64", "float64", "number"]).columns
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns

print("Numeric columns:", list(numeric_features))
print("Categorical columns:", list(categorical_features))

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # keep sparse default
])

## Combine preprocessing
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
log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
])

log_reg.fit(X_train, y_train)

print("Train accuracy:", log_reg.score(X_train, y_train))
print("Test accuracy:", log_reg.score(X_test, y_test))

### 9.1.2 Random Forest

rf = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

rf.fit(X_train, y_train)

print("Train accuracy:", rf.score(X_train, y_train))
print("Test accuracy:", rf.score(X_test, y_test))

## 9.2 Round 2: Addressing Class Imbalance

### 9.2.1 Logistic Regression

log_reg_2 = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
]).fit(X_train, y_train)

log_reg_2.fit(X_train, y_train)

print("Train accuracy:", log_reg_2.score(X_train, y_train))
print("Test accuracy:", log_reg_2.score(X_test, y_test))

### 9.2.2 Random Forest

rf_2 = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
]).fit(X_train, y_train)

rf_2.fit(X_train, y_train)

print("Train accuracy:", rf_2.score(X_train, y_train))
print("Test accuracy:", rf_2.score(X_test, y_test))

# 10. Model Comparison

"""
Evaluate models with multiple metrics:

- ROC AUC quantifies model discrimination capability.
- Precision, recall, F1 reflect correctness and completeness of anomaly identification.
- Average precision exemplifies performance under class imbalance (like PR AUC).
- Balanced accuracy balances sensitivity and specificity.
- Classification report and confusion matrix detail model outputs per class.
"""

## 10.1 Metrics: ROC, AUC, Precision, Recall, Average Precision, F1-Score, Balanced Accuracy
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }
    print(metrics)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics

## 10.2 Model Evaluation

### 10.2.1 Logistic Regression
metrics_lr = evaluate(log_reg, X_test, y_test)
metrics_lr_2 = evaluate(log_reg_2, X_test, y_test)

### 10.2.2 Random Forest
metrics_rf = evaluate(rf, X_test, y_test)
metrics_rf_2 = evaluate(rf_2, X_test, y_test)

summary = pd.DataFrame([
    {"model": "Logistic Regression (Before Class Imbalance)", **metrics_lr},
    {"model": "Logistic Regression (After Class Imbalance)", **metrics_lr_2},
    {"model": "Random Forest (Before Class Imbalance)", **metrics_rf},
    {"model": "Random Forest (After Class Imbalance)", **metrics_rf_2},
])
print(summary)

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

'''

# 11. Next steps for Anomaly Detection

"""
Threshold tuning for maximized F1 and cost-sensitive optimization:

- F1-based threshold balances precision/recall tradeoff tailoring model to business preferences.
- Cost-sensitive thresholding incorporates real-world impact costs of errors, improving practical utility.
- Visualization of F1 and cost curves assist decision makers selecting the operational point.
"""

## 11.1 F1-Max thresholding
y_scores = log_reg.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_scores)

### Compute F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

### Align (thresholds shorter by 1 than precisions/recalls)
best_idx = np.argmax(f1_scores[:-1])
best_threshold_f1 = thresholds_pr[best_idx]
best_f1 = f1_scores[best_idx]

print(f"[F1] Best threshold: {best_threshold_f1:.3f}, F1={best_f1:.3f}")

### Apply best F1 threshold
y_pred_f1 = (y_scores >= best_threshold_f1).astype(int)

## 11.2 Cost-sensitive threshold
## Tunable costs
C_FN = 10  # Cost of a false negative
C_FP = 1   # Cost of a false positive

### Compute ROC curve values
fpr, tpr, thresholds_roc = roc_curve(y_test, y_scores)

n_pos = sum(y_test)
n_neg = len(y_test) - n_pos

costs = []
for thr, fpr_val, tpr_val in zip(thresholds_roc, fpr, tpr):
    FN = (1 - tpr_val) * n_pos  # Expected number of false negatives at this threshold
    FP = fpr_val * n_neg        # Expected number of false positives at this threshold
    cost = C_FN * FN + C_FP * FP
    costs.append((thr, cost))

### Find best threshold minimizing expected cost
best_thr_cost, min_cost = min(costs, key=lambda x: x[1])
print(f"[COST] Best threshold: {best_thr_cost:.3f}, Cost={min_cost:.2f}")

### Apply threshold to predicted scores to get binary predictions
y_pred_cost = (y_scores >= best_thr_cost).astype(int)

### Data Viz
plt.figure(figsize=(10,4))

# F1 Score vs Threshold
plt.subplot(1, 2, 1)
plt.plot(thresholds_pr, f1_scores[:-1], label="F1 score")
plt.axvline(best_threshold_f1, color="r", linestyle="--", label=f"Best F1={best_f1:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold vs F1 Score")
plt.legend()

# Cost vs Threshold
threshold_list = [thr for thr, cost_val in costs]
cost_list = [cost_val for thr, cost_val in costs]
min_cost_idx = np.argmin(cost_list)
best_thr_cost = threshold_list[min_cost_idx]
min_cost = cost_list[min_cost_idx]

plt.subplot(1, 2, 2)
plt.plot(threshold_list, cost_list, label="Expected Cost")
plt.axvline(best_thr_cost, color="r", linestyle="--", linewidth=2, label=f"Best Cost={min_cost:.2f}")
plt.scatter([best_thr_cost], [min_cost], color="r", zorder=5)
plt.xlabel("Threshold")
plt.ylabel("Expected Cost")
plt.title("Threshold vs Cost")
plt.legend()

plt.tight_layout()
plt.show()


# 12. Deployment and Monitoring

"""
Model monitoring and drift detection for production robustness:

- Prevalence drift detection identifies changes in anomaly base rate affecting model performance.
- Feature drift tests (KS-test, Chi2) detect input distribution shifts signaling retraining needs.
- Calibration drift visualization (reliability plot and Brier score) ensures output probabilities remain meaningful over time.
"""

## 12.1 Prevalence Drift
train_prev = y_train.mean()
test_prev = y_test.mean()

count = np.array([y_train.sum(), y_test.sum()])
nobs = np.array([len(y_train), len(y_test)])

z_stat, p_val = proportions_ztest(count, nobs)
print(f"[Prevalence Drift] train={train_prev:.3f}, test={test_prev:.3f}, z={z_stat:.3f}, p={p_val:.3f}")

### Effect size: absolute difference
print(f"Prevalence absolute drift = {abs(train_prev - test_prev):.3f}")

### Data Viz
# Prevalence drift plot
plt.bar(['Train', 'Test'], [train_prev, test_prev])
plt.ylabel('Prevalence of Anomaly')
plt.title('Prevalence Drift')
plt.show()

## 12.2 Feature Drift
print("\n[Feature Drift Tests]")
drift_report = []

for col in X_train.columns:
    if X_train[col].dtype.kind in "bifc":  # numeric
        stat, p_val = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
        drift_report.append((col, "KS-test", p_val))
    else:  # categorical
        train_counts = X_train[col].value_counts(normalize=True)
        test_counts = X_test[col].value_counts(normalize=True)
        categories = set(train_counts.index).union(set(test_counts.index))
        table = [
            [train_counts.get(cat, 0), test_counts.get(cat, 0)]
            for cat in categories
        ]
        chi2, p_val, _, _ = chi2_contingency(table)
        drift_report.append((col, "Chi2-test", p_val))

### Summarize results
df_drift = pd.DataFrame(drift_report, columns=["Feature", "Test", "p-value"])
df_drift["drift_flag"] = df_drift["p-value"] < 0.05
print(df_drift.sort_values("p-value").head(10))

### Data Viz
# Feature drift p-value distribution
plt.hist(df_drift['p-value'], bins=20)
plt.axvline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.xlabel('p-value')
plt.ylabel('Count')
plt.title('Feature Drift Test p-value Distribution')
plt.legend()
plt.show()

### 12.3 Calibration Drift
brier = brier_score_loss(y_test, y_scores)
print(f"\n[Calibration Drift] Brier score: {brier:.3f}")

prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Observed")
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.title("Calibration curve")
plt.legend()
plt.show()
'''