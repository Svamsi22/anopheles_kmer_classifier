"""
step2_build_dataset.py

"""

import os
import pandas as pd

from config import (
    SPECIES_REGISTRY,
    FINAL_TRAIN, FINAL_VAL, FINAL_TEST,
)

# Split ratios — must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = remainder (0.15)


def load_all_species() -> pd.DataFrame:
    
    frames = []
    for sp in SPECIES_REGISTRY:
        path = f"data/raw/kmer_{sp['name']}.csv"
        if not os.path.exists(path):
            print(f"  [WARNING] {path} not found — skipping {sp['name']}.")
            continue
        df = pd.read_csv(path)
        frames.append(df)
        print(f"  Loaded {sp['name']:6s} : {len(df)} samples")

    return pd.concat(frames, ignore_index=True)



def balance(df: pd.DataFrame) -> pd.DataFrame:
    
    counts    = df["species"].value_counts()
    min_count = counts.min()

    print(f"\n  Samples before balancing:")
    print(counts.to_string())
    print(f"\n  Under-sampling all species to {min_count} samples each ...")

    balanced = (
        df
        .groupby("species", group_keys=False)
        .apply(lambda grp: grp.sample(min_count, random_state=42))
        .sample(frac=1, random_state=42)   # global shuffle
        .reset_index(drop=True)
    )

    print(f"\n  Samples after balancing:")
    print(balanced["species"].value_counts().to_string())

    return balanced




def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_frames, val_frames, test_frames = [], [], []

    for species, group in df.groupby("species"):
        group      = group.sample(frac=1, random_state=42)  # shuffle within species
        n          = len(group)
        train_end  = int(TRAIN_RATIO * n)
        val_end    = train_end + int(VAL_RATIO * n)

        train_frames.append(group.iloc[:train_end])
        val_frames.append(  group.iloc[train_end:val_end])
        test_frames.append( group.iloc[val_end:])

    def _concat_shuffle(frames):
        return (
            pd.concat(frames, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    return (
        _concat_shuffle(train_frames),
        _concat_shuffle(val_frames),
        _concat_shuffle(test_frames),
    )




def save(train_df, val_df, test_df):
    """Save the three final CSVs and print a summary."""
    for path in [FINAL_TRAIN, FINAL_VAL, FINAL_TEST]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    train_df.to_csv(FINAL_TRAIN, index=False)
    val_df.to_csv(FINAL_VAL,     index=False)
    test_df.to_csv(FINAL_TEST,   index=False)

    print("\n" + "="*55)
    print("  Final datasets saved")
    print("="*55)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {name} — {len(df)} samples")
        print(df["species"].value_counts().to_string())




def main():
    print("="*55)
    print("  Building Final Datasets")
    print("="*55)

    print("\nLoading raw species data ...")
    raw = load_all_species()

    print("\nBalancing classes ...")
    balanced = balance(raw)

    print("\nSplitting (stratified per species) ...")
    train_df, val_df, test_df = stratified_split(balanced)

    save(train_df, val_df, test_df)
    print("\nNext step: run step3_explore.py")


if __name__ == "__main__":
    main()
