import os
import json
from glob import glob

import numpy as np
import pandas as pd
import joblib
import kagglehub

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Config
# -----------------------------
DATASET_ID = "loveall/appliances-energy-prediction"
TARGET_COL = "Appliances"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "energy_rf_pipeline.joblib")


# -----------------------------
# Helpers
# -----------------------------
def download_and_load() -> pd.DataFrame:
    """
    Downloads the latest dataset version using kagglehub and loads the CSV.
    """
    path = kagglehub.dataset_download(DATASET_ID)
    csv_files = glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    # This dataset usually contains KAG_energydata_complete.csv
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option B: API will send a date string.
    We compute engineered time features during training in the same way,
    and later we will compute them inside the API as well.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid dates (rare, but safe)
    df = df.dropna(subset=["date"])

    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month_name()
    df["weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # We won't train on raw date
    df = df.drop(columns=["date"])
    return df


def build_pipeline():
    """
    Builds preprocessing + model pipeline.
    Matches your notebook feature groups.
    """
    numeric_features = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility",
        "Tdewpoint", "hour", "day"
    ]
    categorical_features = ["day_of_week", "month"]
    binary_features = ["weekend"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean"))
            ]), numeric_features),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),

            ("bin", "passthrough", binary_features),
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", rf)
    ])
    return pipeline


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def stratified_split_regression(df: pd.DataFrame, target_col: str, test_size: float = 0.2, q: int = 5):
    """
    Stratified split for regression by binning the target into quantiles.
    Mirrors your notebook approach with StratifiedShuffleSplit + pd.qcut.
    """
    df_for_split = df.copy()

    # Create bins for stratification (quantiles)
    df_for_split["Appliance_Bins"] = pd.qcut(
        df_for_split[target_col],
        q=q,
        labels=False,
        duplicates="drop"
    )

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    for train_idx, test_idx in split.split(df_for_split, df_for_split["Appliance_Bins"]):
        train_set = df_for_split.iloc[train_idx].drop(columns=["Appliance_Bins"])
        test_set = df_for_split.iloc[test_idx].drop(columns=["Appliance_Bins"])

    X_train = train_set.drop(columns=[target_col])
    y_train = train_set[target_col]
    X_test = test_set.drop(columns=[target_col])
    y_test = test_set[target_col]

    return X_train, X_test, y_train, y_test


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load + feature engineering
    df = download_and_load()
    df = add_time_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

    # Stratified split (regression)
    X_train, X_test, y_train, y_test = stratified_split_regression(df, TARGET_COL, test_size=0.2, q=5)

    # Recommended prints: verify stratification worked (great for report screenshots)
    print("Train distribution (5 quantiles):\n", pd.qcut(y_train, q=5).value_counts(normalize=True))
    print("\nTest distribution (5 quantiles):\n", pd.qcut(y_test, q=5).value_counts(normalize=True))

    # Train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)

    print("\nTrain metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # Save model artifact (ONE file: preprocessing + model)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline to: {MODEL_PATH}")

    # Save metrics to include in report
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
