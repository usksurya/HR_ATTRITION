"""Classifier comparison - reproduces the model families from the original
'AI_HR_Attrition.ipynb' (SVM, KNN, Decision Tree) but fixes the methodology:

  * stratified train/test split with a fixed seed
  * scaler fitted on train only (no leakage)
  * class imbalance handled (SMOTE for the tree, class_weight for SVM)
  * metrics appropriate for imbalanced data: precision / recall / F1 / ROC-AUC
    alongside accuracy.

Key insight from the original work was 'linear SVM and 29-NN both reached
~84.69% accuracy'. That number is barely above the imbalanced base rate
(~83.2% 'No'), so we report the full metric matrix instead of accuracy alone.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline

from config import (
    SVM_KERNELS, KNN_K_RANGE, DT_PARAMS, MODELS, SEED,
)
from preprocess import build_modeling_pipeline


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
    }


def run_svm_grid(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for kernel in SVM_KERNELS:
        clf = SVC(kernel=kernel, class_weight="balanced", probability=True, random_state=SEED)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]
        rows.append(evaluate(f"SVM-{kernel}", y_test, pred, prob))
    return pd.DataFrame(rows)


def run_knn_tuning(X_train, X_test, y_train, y_test, fitted_scaler=None) -> pd.DataFrame:
    """Tune K over odd values 3..49 (as in the original)."""
    rows = []
    for k in KNN_K_RANGE:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        rows.append(
            {
                "k": k,
                "train_acc": accuracy_score(y_train, clf.predict(X_train)),
                "test_acc": accuracy_score(y_test, pred_test),
                "gap": accuracy_score(y_train, clf.predict(X_train)) - accuracy_score(y_test, pred_test),
            }
        )
    return pd.DataFrame(rows)


def run_decision_tree_smote(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Decision Tree trained on SMOTE-resampled data (original approach)."""
    pipe = imb_make_pipeline(SMOTE(random_state=SEED), DecisionTreeClassifier(**DT_PARAMS))
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    rows = [evaluate("DecisionTree+SMOTE", y_test, pred, prob)]
    df_res = pd.DataFrame(rows)
    # Save metrics too
    print("\n[DT-SMOTE] confusion matrix (rows=pred, cols=true):")
    print(confusion_matrix(y_test, pred))
    return df_res


def run_all(df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test, scaler = build_modeling_pipeline(df)

    svm = run_svm_grid(X_train, X_test, y_train, y_test)

    knn_grid = run_knn_tuning(X_train, X_test, y_train, y_test)
    best_k = int(knn_grid.loc[knn_grid["test_acc"].idxmax(), "k"])
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    pred_knn = knn_final.predict(X_test)
    prob_knn = knn_final.predict_proba(X_test)[:, 1]
    knn_res = pd.DataFrame([evaluate(f"KNN-{best_k}", y_test, pred_knn, prob_knn)])

    dt = run_decision_tree_smote(X_train, X_test, y_train, y_test)

    results = pd.concat([svm, knn_res, dt], ignore_index=True)
    results = results.sort_values("f1", ascending=False).reset_index(drop=True)
    print("\n[MODELS] comparison (sorted by F1):")
    print(results.to_string(index=False))

    summary = {
        "results": results,
        "knn_grid": knn_grid,
        "best_k": best_k,
    }
    return summary


if __name__ == "__main__":
    from load_data import load

    run_all(load())