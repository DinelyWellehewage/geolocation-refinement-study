import pandas as pd
from src.datasets.folds import create_fold_split


def create_fold_split(
    folds_csv,
    fold,
    train_output,
    val_output,
):
    df = pd.read_csv(folds_csv)

    train_df = df[df["fold"] != fold].copy()
    val_df = df[df["fold"] == fold].copy()

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"Fold: {fold}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")