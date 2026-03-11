"""
step4_explore.py

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

from config import (
    META_COLS, SPECIES_COLORS,
    FINAL_TRAIN,
    FIG_PCA, FIG_TSNE,
)



def load_data():
    
    df       = pd.read_csv(FINAL_TRAIN)
    kmer_cols = [c for c in df.columns if c not in META_COLS]
    X        = df[kmer_cols].values
    y        = df["species"].values
    labels   = df["label"].values
    return df, X, y, labels, kmer_cols




def _scatter_by_species(ax, coords, y, title, xlabel, ylabel):
    
    for species, color in SPECIES_COLORS.items():
        mask = y == species
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=species, color=color,
            alpha=0.7, s=25, linewidths=0,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Species", fontsize=9)




def plot_pca(X: np.ndarray, y: np.ndarray):
    
    print("Running PCA ...")
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_by_species(
        ax, X_pca, y,
        title="K-mer PCA — 2 Principal Components",
        xlabel=f"PC1  ({var[0]:.1f}% variance)",
        ylabel=f"PC2  ({var[1]:.1f}% variance)",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_PCA), exist_ok=True)
    plt.savefig(FIG_PCA, dpi=150)
    plt.show()
    print(f"  Saved → {FIG_PCA}")


def plot_tsne(X: np.ndarray, y: np.ndarray, perplexity: int = 30):
    
    print("Running t-SNE (this may take a minute) ...")
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_by_species(
        ax, X_tsne, y,
        title=f"K-mer t-SNE  (perplexity={perplexity})",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    plt.tight_layout()
    fig_path = FIG_TSNE
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"  Saved → {fig_path}")


def plot_gc_content(df: pd.DataFrame, kmer_cols: list[str]):
    
    print("Computing GC content ...")

    gc_kmers = [k for k in kmer_cols if all(b in "GC" for b in k)]
    df       = df.copy()
    df["gc_content"] = df[gc_kmers].sum(axis=1)

    gc_means = df.groupby("species")["gc_content"].mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    colors  = [SPECIES_COLORS.get(sp, "gray") for sp in gc_means.index]
    bars    = ax.bar(gc_means.index, gc_means.values, color=colors,
                     edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, gc_means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Mean GC Content by Species", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean GC k-mer frequency")
    ax.set_xlabel("Species")

    plt.tight_layout()
    fig_path = "outputs/figures/gc_content.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"  Saved → {fig_path}")
    print("\nGC content per species:")
    print(gc_means.to_string())


def plot_rf_importances(X: np.ndarray, labels: np.ndarray, kmer_cols: list[str],
                        top_n: int = 20):
    
    print(f"\nFitting Random Forest for feature importances (top {top_n}) ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                n_jobs=-1, random_state=42)
    rf.fit(X, labels)

    imp_df = (
        pd.DataFrame({"kmer": kmer_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    print(f"\nTop {top_n} k-mers by RF importance:")
    print(imp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["kmer"][::-1], imp_df["importance"][::-1],
            color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} K-mers — Random Forest Importance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("K-mer")

    plt.tight_layout()
    fig_path = "outputs/figures/rf_importances.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"  Saved → {fig_path}")




def main():
    print("="*55)
    print("  Exploratory Data Analysis")
    print("="*55)

    df, X, y, labels, kmer_cols = load_data()

    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("Class distribution:")
    print(pd.Series(y).value_counts().to_string())

    plot_pca(X, y)
    plot_tsne(X, y)
    plot_gc_content(df, kmer_cols)
    plot_rf_importances(X, labels, kmer_cols)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
