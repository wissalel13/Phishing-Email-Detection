"""
Split générique train/test stratifié pour un CSV labellisé.

Usage:
    python -m src.data.split_dataset <input.csv> <label_col> <output_prefix> [--test-size 0.2]
"""

import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split(input_csv: str, label_col: str, output_prefix: str, test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(input_csv)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],   # préserve le ratio de classes dans train ET test
        random_state=seed,        # reproductibilité
    )

    train_path = f"{output_prefix}_train.csv"
    test_path = f"{output_prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Total: {len(df)} lignes")
    print(f"Train: {len(train_df)} lignes -> {train_path}")
    print(f"Test:  {len(test_df)} lignes -> {test_path}")
    print("\nRépartition des classes (train):")
    print(train_df[label_col].value_counts(normalize=True).round(3))
    print("\nRépartition des classes (test):")
    print(test_df[label_col].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("label_col")
    parser.add_argument("output_prefix")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    split(args.input_csv, args.label_col, args.output_prefix, args.test_size)
