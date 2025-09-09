# models/evaluate.py

from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
import logging

def evaluate(model, X_train, X_test, y_train, y_test):
    logging.info("Evaluating model performance")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }

    logging.info(f"Evaluation metrics: {metrics}")
    logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics
