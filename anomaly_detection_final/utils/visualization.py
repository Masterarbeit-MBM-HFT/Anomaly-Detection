# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y_train, y_test):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.countplot(x=y_train)
    plt.title('Train Set Class Distribution')
    plt.subplot(1,2,2)
    sns.countplot(x=y_test)
    plt.title('Test Set Class Distribution')
    plt.show()

def plot_roc_curves(models, X_test, y_test):
    from sklearn.metrics import roc_curve, roc_auc_score
    plt.figure(figsize=(8,6))
    for name, model in models:
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()