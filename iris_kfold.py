# iris_kfold.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def quick_cv(X, y, model, n_splits=5, stratified=True, random_state=42):
    """Quick evaluation using cross_val_score (returns array of scores)."""
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores

def detailed_kfold(X, y, model, n_splits=5, stratified=True, random_state=42):
    """Manual K-Fold loop so we can inspect per-fold metrics and confusion matrices."""
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 0
    accuracies = []
    cms = []
    reports = []

    for train_idx, test_idx in kf.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred, zero_division=0)

        print(f"\n--- Fold {fold} ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(rep)

        accuracies.append(acc)
        cms.append(cm)
        reports.append(rep)

    return np.array(accuracies), cms, reports

def plot_fold_accuracies(accs):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(accs)+1), accs, marker='o')
    plt.title("K-Fold Accuracies")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0,1.05)
    plt.grid(True)
    plt.xticks(np.arange(1, len(accs)+1))
    plt.show()

def main():
    # Load Iris
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print("Dataset:", data.DESCR.splitlines()[0])
    print("Features:", feature_names)
    print("Classes:", target_names)
    print("Shape X:", X.shape, " Shape y:", y.shape)

    # Choose model
    model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto', random_state=0)

    # Quick CV (5-fold)
    print("\nRunning quick cross_val_score (5-fold Stratified)...")
    quick_scores = quick_cv(X, y, model, n_splits=5, stratified=True, random_state=42)
    print("Fold accuracies:", np.round(quick_scores, 4))
    print("Mean accuracy:", np.mean(quick_scores).round(4), "Std:", np.std(quick_scores).round(4))

    # Detailed K-Fold
    print("\nRunning detailed K-Fold (manual loop)...")
    accs, cms, reports = detailed_kfold(X, y, model, n_splits=5, stratified=True, random_state=42)
    print("\nAll fold accuracies:", np.round(accs, 4))
    print("Mean accuracy (manual):", np.mean(accs).round(4))

    # Plot fold accuracies
    plot_fold_accuracies(accs)

if __name__ == "__main__":
    main()
