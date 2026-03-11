"""
step5_train_evaluate.py

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from config import (
    META_COLS, SPECIES_NAMES, SPECIES_COLORS,
    FINAL_TRAIN, FINAL_VAL, FINAL_TEST,
    FIG_CONFUSION, FIG_COEF,
)



def load_splits():
    
    train = pd.read_csv(FINAL_TRAIN)
    val   = pd.read_csv(FINAL_VAL)
    test  = pd.read_csv(FINAL_TEST)

    kmer_cols = [c for c in train.columns if c not in META_COLS]

    print("Shapes:")
    print(f"  Train : {train.shape}")
    print(f"  Val   : {val.shape}")
    print(f"  Test  : {test.shape}")

    print("\nClass distribution (train):")
    print(train.groupby(["species", "label"]).size().to_string())

    return (
        train[kmer_cols], train["label"],
        val[kmer_cols],   val["label"],
        test[kmer_cols],  test["label"],
        kmer_cols,
    )



def scale(X_train, X_val, X_test):
    
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_val_s    = scaler.transform(X_val)
    X_test_s   = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler




def build_model() -> LogisticRegression:
    
    return LogisticRegression(
        C=0.1,
        max_iter=1_000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )



def print_reports(model, X_val_s, y_val, X_test_s, y_test):
    print("\n" + "="*55)
    print("  Validation Results")
    print("="*55)
    print(classification_report(
        y_val, model.predict(X_val_s),
        target_names=SPECIES_NAMES,
    ))

    print("="*55)
    print("  Test Results (unseen data)")
    print("="*55)
    print(classification_report(
        y_test, model.predict(X_test_s),
        target_names=SPECIES_NAMES,
    ))


def run_cross_validation(model, X_train_s, y_train):
    
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_s, y_train,
                             cv=cv, scoring="f1_macro")
    print(f"\n5-Fold CV  F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")
    return cv


def print_learning_curve(model, X_train_s, y_train, cv):
    
    train_sizes, _, val_scores = learning_curve(
        build_model(),           
        X_train_s, y_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    print("\nLearning curve (training size → CV F1):")
    for size, mean_f1 in zip(train_sizes, val_scores.mean(axis=1)):
        print(f"  {int(size):>4} samples → {mean_f1:.3f}")




def plot_confusion_matrices(model, X_val_s, y_val, X_test_s, y_test):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, X_s, y, title in [
        (axes[0], X_val_s,  y_val,  "Validation Confusion Matrix"),
        (axes[1], X_test_s, y_test, "Test Confusion Matrix"),
    ]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_s, y,
            display_labels=SPECIES_NAMES,
            ax=ax, colorbar=False,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_CONFUSION), exist_ok=True)
    plt.savefig(FIG_CONFUSION, dpi=150)
    plt.show()
    print(f"  Saved → {FIG_CONFUSION}")


def plot_kmer_coefficients(model, kmer_cols: list[str], top_n: int = 10):
    
    print(f"\nTop {top_n} discriminative k-mers per species:")

    fig, axes = plt.subplots(1, len(SPECIES_NAMES),
                             figsize=(6 * len(SPECIES_NAMES), 6))

    for class_idx, species in enumerate(SPECIES_NAMES):
        coef_series = (
            pd.DataFrame({
                "kmer":        kmer_cols,
                "coefficient": model.coef_[class_idx],
            })
            .sort_values("coefficient", ascending=False)
            .head(top_n)
        )

        print(f"\n  {species}:")
        print(coef_series.to_string(index=False))

        ax    = axes[class_idx]
        color = SPECIES_COLORS.get(species, "steelblue")
        ax.barh(coef_series["kmer"], coef_series["coefficient"],
                color=color, edgecolor="white")
        ax.set_title(f"Top K-mers → {species}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Logistic Regression Coefficient")
        ax.invert_yaxis()

    plt.suptitle("Most Discriminative K-mers Per Species",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_COEF), exist_ok=True)
    plt.savefig(FIG_COEF, dpi=150)
    plt.show()
    print(f"  Saved → {FIG_COEF}")



def main():
    print("="*55)
    print("  Training Logistic Regression Classifier")
    print("="*55)

    
    X_train, y_train, X_val, y_val, X_test, y_test, kmer_cols = load_splits()

    
    X_train_s, X_val_s, X_test_s, _ = scale(X_train, X_val, X_test)

    model = build_model()
    model.fit(X_train_s, y_train)
    print("\nModel trained.")

    print_reports(model, X_val_s, y_val, X_test_s, y_test)
    cv = run_cross_validation(model, X_train_s, y_train)
    print_learning_curve(model, X_train_s, y_train, cv)

    # Figures
    plot_confusion_matrices(model, X_val_s, y_val, X_test_s, y_test)
    plot_kmer_coefficients(model, kmer_cols)

    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
