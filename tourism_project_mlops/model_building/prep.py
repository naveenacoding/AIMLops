from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"
OUTPUT_DIR = ROOT / "artifacts"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove identifier/index columns that should not be model features.
    df = df.drop(
        columns=["CustomerID", "Unnamed: 0"],
        errors="ignore",
    )

    if "ProdTaken" not in df.columns:
        raise ValueError("Missing target column: ProdTaken")

    df = df.dropna(subset=["ProdTaken"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["ProdTaken"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Data preparation completed")
    print("Training samples:", len(train_df))
    print("Testing samples :", len(test_df))
    print("Saved:", train_path)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()
