"""
step6_compare_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import (
    META_COLS, SPECIES_NAMES,
    FINAL_TRAIN, FINAL_VAL, FINAL_TEST,
    FIG_COMPARISON,
)


def load_splits():
    train = pd.read_csv(FINAL_TRAIN, low_memory=False)
    val   = pd.read_csv(FINAL_VAL,   low_memory=False)
    test  = pd.read_csv(FINAL_TEST,  low_memory=False)

    kmer_cols = [c for c in train.columns if c not in META_COLS]

    return (
        train[kmer_cols].values, train["label"].values,
        val[kmer_cols].values,   val["label"].values,
        test[kmer_cols].values,  test["label"].values,
        kmer_cols,
    )


def gc_feature(df: pd.DataFrame, kmer_cols: list[str]) -> np.ndarray:
    gc_kmers = [k for k in kmer_cols if all(b in "GC" for b in k)]
    return df[gc_kmers].sum(axis=1).values.reshape(-1, 1)


def build_experiments(
    X_tr_km, X_val_km, X_test_km,
    X_tr_gc, X_val_gc, X_test_gc,
    X_tr_comb, X_val_comb, X_test_comb,
):

    return [

        {
            "name": "Random\nBaseline",
            "model": DummyClassifier(strategy="most_frequent"),
            "X_train": X_tr_km,
            "X_val": X_val_km,
            "X_test": X_test_km,
            "color": "#d9534f",
            "note": "Predicts majority class — no learning.",
        },

        {
            "name": "GC Content\nOnly",
            "model": LogisticRegression(C=0.1, max_iter=1000),
            "X_train": X_tr_gc,
            "X_val": X_val_gc,
            "X_test": X_test_gc,
            "color": "#f0ad4e",
            "note": "Single feature: aggregate GC frequency.",
        },

        {
            "name": "LogReg\n(kmers)",
            "model": LogisticRegression(
                C=0.1,
                max_iter=1000,
                multi_class="multinomial"
            ),
            "X_train": X_tr_km,
            "X_val": X_val_km,
            "X_test": X_test_km,
            "color": "#2ecc71",
            "note": "Logistic regression using k-mer features.",
        },

        {
            "name": "Random Forest\n(kmers)",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            ),
            "X_train": X_tr_km,
            "X_val": X_val_km,
            "X_test": X_test_km,
            "color": "#5bc0de",
            "note": "Random forest using k-mer features.",
        },

        {
            "name": "LogReg\n(kmers + GC)",
            "model": LogisticRegression(
                C=0.1,
                max_iter=1000,
                multi_class="multinomial"
            ),
            "X_train": X_tr_comb,
            "X_val": X_val_comb,
            "X_test": X_test_comb,
            "color": "#1abc9c",
            "note": "Logistic regression using k-mers + GC content.",
        },

    ]


def run_experiments(experiments, y_train, y_val, y_test):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for exp in experiments:

        m = exp["model"]
        Xtr, Xv, Xt = exp["X_train"], exp["X_val"], exp["X_test"]

        m.fit(Xtr, y_train)

        val_acc = accuracy_score(y_val, m.predict(Xv))
        test_acc = accuracy_score(y_test, m.predict(Xt))

        val_f1 = f1_score(y_val, m.predict(Xv), average="macro", zero_division=0)
        test_f1 = f1_score(y_test, m.predict(Xt), average="macro", zero_division=0)

        cv_scores = cross_val_score(
            m, Xtr, y_train, cv=cv, scoring="f1_macro"
        )

        record = {
            "name": exp["name"],
            "note": exp["note"],
            "color": exp["color"],
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        results.append(record)

        label = exp["name"].replace("\n", " ")

        print("\n" + "=" * 45)
        print(f"  {label}")
        print(f"  Note    : {exp['note']}")
        print("=" * 45)
        print(f"  Val  Accuracy : {val_acc:.3f}")
        print(f"  Test Accuracy : {test_acc:.3f}")
        print(f"  Val  F1 Macro : {val_f1:.3f}")
        print(f"  Test F1 Macro : {test_f1:.3f}")
        print(
            f"  CV   F1 Macro : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

    return results


def plot_comparison(results):

    names = [r["name"] for r in results]
    colors = [r["color"] for r in results]
    x = np.arange(len(results))
    w = 0.55

    CHANCE = 1 / len(SPECIES_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    fig.suptitle(
        "K-mer Features vs Baselines — 3-Species Anopheles Classification",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    panels = [

        ("Validation Accuracy",
         [r["val_acc"] for r in results], None),

        ("Test F1 Score (Macro)",
         [r["test_f1"] for r in results], None),

        ("Cross-Val F1 ± Std\n(5-fold, train)",
         [r["cv_mean"] for r in results],
         [r["cv_std"] for r in results]),
    ]

    for ax, (title, values, errors) in zip(axes, panels):

        bar_kw = dict(color=colors, edgecolor="white", linewidth=1.5)

        if errors:
            bars = ax.bar(
                x, values, w,
                yerr=errors,
                capsize=6,
                error_kw={"linewidth": 2, "color": "black"},
                **bar_kw,
            )
        else:
            bars = ax.bar(x, values, w, **bar_kw)

        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.axhline(
            CHANCE,
            color="red",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
            label=f"Random chance ({CHANCE:.2f})",
        )

        ax.legend(fontsize=8)

        for bar, val, err in zip(bars, values, errors or [0] * len(values)):

            label = f"{val:.3f}\n±{err:.3f}" if errors else f"{val:.3f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (err or 0) + 0.03,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    patches = [
        mpatches.Patch(
            color=r["color"],
            label=r["name"].replace("\n", " "),
        )
        for r in results
    ]

    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        edgecolor="gray",
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(FIG_COMPARISON), exist_ok=True)

    plt.savefig(FIG_COMPARISON, dpi=150, bbox_inches="tight")

    plt.show()

    print(f"\nSaved → {FIG_COMPARISON}")


def print_summary_table(results):

    print("\n" + "=" * 65)
    print(f"{'Model':<28} {'Val Acc':>8} {'Test F1':>8} {'CV F1':>8} {'CV Std':>8}")
    print("=" * 65)

    for r in results:

        name = r["name"].replace("\n", " ")

        print(
            f"{name:<28} {r['val_acc']:>8.3f} {r['test_f1']:>8.3f} "
            f"{r['cv_mean']:>8.3f} {r['cv_std']:>8.3f}"
        )

    print("=" * 65)


def main():

    print("=" * 55)
    print("  Model Comparison")
    print("=" * 55)

    X_tr_km, y_train, X_val_km, y_val, X_test_km, y_test, kmer_cols = load_splits()

    scaler = StandardScaler()

    X_tr_km = scaler.fit_transform(X_tr_km)
    X_val_km = scaler.transform(X_val_km)
    X_test_km = scaler.transform(X_test_km)

    train_df = pd.read_csv(FINAL_TRAIN, low_memory=False)
    val_df = pd.read_csv(FINAL_VAL, low_memory=False)
    test_df = pd.read_csv(FINAL_TEST, low_memory=False)

    X_tr_gc = gc_feature(train_df, kmer_cols)
    X_val_gc = gc_feature(val_df, kmer_cols)
    X_test_gc = gc_feature(test_df, kmer_cols)

    gc_scaler = StandardScaler()

    X_tr_gc = gc_scaler.fit_transform(X_tr_gc)
    X_val_gc = gc_scaler.transform(X_val_gc)
    X_test_gc = gc_scaler.transform(X_test_gc)

    X_tr_comb = np.hstack([X_tr_km, X_tr_gc])
    X_val_comb = np.hstack([X_val_km, X_val_gc])
    X_test_comb = np.hstack([X_test_km, X_test_gc])

    experiments = build_experiments(
        X_tr_km, X_val_km, X_test_km,
        X_tr_gc, X_val_gc, X_test_gc,
        X_tr_comb, X_val_comb, X_test_comb,
    )

    results = run_experiments(experiments, y_train, y_val, y_test)

    print_summary_table(results)

    plot_comparison(results)

    print("\nModel comparison complete.")


if __name__ == "__main__":
    main()