from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DEPLOYMENT = ROOT / "deployment"


def main():
    train_path = ARTIFACTS / "train.csv"
    test_path = ARTIFACTS / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train/test files are missing. Run model_building/prep.py first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=["ProdTaken"])
    y_train = train["ProdTaken"].astype(int)

    X_test = test.drop(columns=["ProdTaken"])
    y_test = test["ProdTaken"].astype(int)

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [3, 4],
        "classifier__learning_rate": [0.05, 0.10],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    tracking_dir = ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("tourism_prediction")

    with mlflow.start_run():
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        DEPLOYMENT.mkdir(parents=True, exist_ok=True)

        model_path = DEPLOYMENT / "model.joblib"
        joblib.dump(best_model, model_path)

        defaults = {}
        categories = {}

        for column in X_train.columns:
            if column in numeric:
                median_value = X_train[column].median()
                defaults[column] = 0.0 if pd.isna(median_value) else float(median_value)
            else:
                values = X_train[column].dropna().astype(str)
                defaults[column] = str(values.mode().iloc[0]) if not values.empty else ""
                categories[column] = sorted(values.unique().tolist())

        schema = {
            "columns": X_train.columns.tolist(),
            "numeric": numeric,
            "categorical": categorical,
            "defaults": defaults,
            "categories": categories,
        }

        schema_path = DEPLOYMENT / "feature_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(schema_path), artifact_path="model")

        print("Model training completed")
        print("Best parameters:", search.best_params_)
        print("Accuracy:", round(accuracy, 4))
        print("F1 Score:", round(f1, 4))
        print(classification_report(y_test, predictions, zero_division=0))
        print("Model saved:", model_path)
        print("Schema saved:", schema_path)


if __name__ == "__main__":
    main()
