from pathlib import Path
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import skops.io as sio

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "drug.csv"
MODEL_DIR = ROOT / "Model"
RESULTS_DIR = ROOT / "Results"
TARGET = "Drug"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Drug Classification dataset, rename it drug.csv, and place it in Data/."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}'. Found columns: {list(df.columns)}"
        )
    if len(df) < 10:
        raise ValueError("Dataset is too small; provide at least 10 rows.")

    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    if not numeric and not categorical:
        raise ValueError("No usable numeric or categorical feature columns were found.")

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    # Stratification is helpful for this dataset, but tiny classes may make it impossible.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sio.dump(pipe, MODEL_DIR / "drug_pipeline.skops")

    metrics = (
        "Drug classification model evaluation\n"
        f"Rows used: {len(df)}\n"
        f"Train rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Features: {', '.join(X.columns)}\n"
        f"Target classes: {', '.join(sorted(y.unique()))}\n"
    )
    (RESULTS_DIR / "metrics.txt").write_text(metrics, encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "rows": int(len(df)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "features": list(X.columns),
                "classes": sorted(y.unique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title("Drug Class Prediction — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_results.png", dpi=160)
    plt.close(fig)

    print(metrics)
    print(f"Saved model: {MODEL_DIR / 'drug_pipeline.skops'}")
    print(f"Saved plot: {RESULTS_DIR / 'model_results.png'}")


if __name__ == "__main__":
    main()
