import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_split(
    csv_file,
    train_output,
    val_output,
    val_size=0.2,
    random_state=42,
):
    df = pd.read_csv(csv_file)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=df["country"],
    )

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    print("\nTraining country distribution:")
    print(train_df["country"].value_counts())

    print("\nValidation country distribution:")
    print(val_df["country"].value_counts())


if __name__ == "__main__":
    create_train_val_split(
        csv_file="geo_dataset/train_labels.csv",
        train_output="geo_dataset/train_split.csv",
        val_output="geo_dataset/val_split.csv",
    )