from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = {
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "ProductPitched",
    "Designation",
    "MonthlyIncome",
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df["ProdTaken"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
