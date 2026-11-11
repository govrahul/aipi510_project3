"""Clean the water-potability data and recreate the notebook's EDA plots.

Run from the repository root with:
    python Data/cleaning_and_EDA.py

The cleaned CSV, feature metadata, and plots are written to Data/eda_output/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent
SOURCE_CSV = DATA_DIR / "water_potability.csv"
OUTPUT_DIR = DATA_DIR / "eda_output"
TARGET = "potability"
LAB_ONLY_COLUMNS = ["Organic_carbon", "Trihalomethanes"]
DROP_TURBIDITY = False
DROP_SOLIDS = True
DATA_SOURCE = (
    "https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability"
)

VALID_RANGES = {
    "ph": (0, 14),
    "hardness": (0, None),
    "solids": (0, None),
    "chloramines": (0, None),
    "sulfate": (0, None),
    "conductivity": (0, None),
    "turbidity": (0, None),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for the cleaned CSV, metadata, and plots (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_and_clean_data():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_CSV}")

    raw = pd.read_csv(SOURCE_CSV)
    columns_to_drop = list(LAB_ONLY_COLUMNS)
    if DROP_TURBIDITY:
        columns_to_drop.append("Turbidity")
    if DROP_SOLIDS:
        columns_to_drop.append("Solids")

    df = raw.drop(
        columns=[column for column in columns_to_drop if column in raw.columns]
    ).copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    features = [column for column in df.columns if column != TARGET]

    non_numeric = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        raise ValueError(f"Unexpected target values: {sorted(target_values)}")
    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    df[TARGET] = df[TARGET].astype(int)

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicate_count} exact duplicate rows.")

    for column, (lower, upper) in VALID_RANGES.items():
        if column not in df.columns:
            continue
        invalid = pd.Series(False, index=df.index)
        if lower is not None:
            invalid |= df[column] < lower
        if upper is not None:
            invalid |= df[column] > upper
        count = int(invalid.sum())
        print(f"{column}: {count} impossible values set to missing.")
        df.loc[invalid, column] = np.nan

    return raw, df, features, columns_to_drop


def save_plots(df, features, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(features), figsize=(3 * len(features), 4))
    for ax, column in zip(np.atleast_1d(axes), features):
        sns.boxplot(y=df[column], ax=ax)
        ax.set_title(column)
    fig.tight_layout()
    boxplot_path = output_dir / "feature_boxplots.png"
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, features):
        sns.kdeplot(data=df, x=column, hue=TARGET, common_norm=False, ax=ax)
        ax.set_title(column)
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    distributions_path = output_dir / "feature_distributions_by_potability.png"
    fig.savefig(distributions_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    correlation_path = output_dir / "correlation_matrix.png"
    fig.savefig(correlation_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return [boxplot_path, distributions_path, correlation_path]


def save_cleaned_data(df, features, columns_to_drop, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "water_potability_clean.csv"
    metadata_path = output_dir / "feature_metadata.json"
    df.to_csv(csv_path, index=False)

    metadata = {
        "source": DATA_SOURCE,
        "target": TARGET,
        "dropped_columns": columns_to_drop,
        "n_rows": int(len(df)),
        "class_balance": {
            str(label): int(count)
            for label, count in df[TARGET].value_counts().items()
        },
        "features": {
            column: {
                "min": round(float(df[column].min()), 3),
                "max": round(float(df[column].max()), 3),
                "median": round(float(df[column].median()), 3),
                "pct_missing": round(float(df[column].isna().mean() * 100), 1),
            }
            for column in features
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return csv_path, metadata_path


def main():
    args = parse_args()
    raw, df, features, columns_to_drop = load_and_clean_data()

    missing = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print("\nMissing values (%):")
    print(missing.to_string())
    print("\nPotability counts:")
    print(df[TARGET].value_counts().sort_index().to_string())

    outlier_counts = {}
    for column in features:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[column] = int(
            ((df[column] < lower) | (df[column] > upper)).sum()
        )
    print("\nIQR outliers:")
    print(pd.Series(outlier_counts, name="IQR outliers").to_string())

    assert not df.duplicated().any()
    assert df.columns.tolist() == features + [TARGET]
    if "ph" in df.columns:
        assert df["ph"].dropna().between(0, 14).all()

    plot_paths = save_plots(df, features, args.output_dir)
    csv_path, metadata_path = save_cleaned_data(
        df, features, columns_to_drop, args.output_dir
    )

    print(f"\nSource rows: {len(raw)}; cleaned rows: {len(df)}")
    print(f"Cleaned data: {csv_path}")
    print(f"Feature metadata: {metadata_path}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
