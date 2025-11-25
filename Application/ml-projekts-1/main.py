from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import os
import warnings
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
dirname = os.path.dirname(__file__)
DATA_PATH = os.path.join(dirname, "data", "smoking_driking_dataset_Ver01.csv")
TARGET = "DRK_YN"
EXPERIMENT_NAME = "Drinking Classification"
REGISTERED_MODEL_NAME = "drk_classifier"
TEST_SIZE = 0.30
RANDOM_STATE = 42

class QuantileClipper(BaseEstimator, TransformerMixin):
    # Outlier clipping
    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_np = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_np, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X_np, self.upper, axis=0)
        return self

    def transform(self, X):
        X_np = np.asarray(X, dtype=float)
        return np.clip(X_np, self.lower_bounds_, self.upper_bounds_)

def load_dataset() -> pd.DataFrame:
    # Nolasa CSV un uzreiz pārvērš kategorijas par stringiem (jo MLflow nepatīk object)
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].astype(str)
    df["SMK_stat_type_cd"] = df["SMK_stat_type_cd"].astype(int).astype(str)
    df["urine_protein"] = df["urine_protein"].astype(int).astype(str)
    df["hear_left"] = df["hear_left"].astype(int).astype(str)
    df["hear_right"] = df["hear_right"].astype(int).astype(str)
    df["sex"] = df["sex"].astype(str)

    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # 70/30 sadalījums 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    # ColumnTransformer veidošana ar atsevišķiem pipeline skaitliskiem/log un kategoriskiem datiem
    numeric_cols = [
        col
        for col in X_train.select_dtypes(include=["int64", "float64"]).columns
        if col not in {"hear_left", "hear_right", "urine_protein", "SMK_stat_type_cd"}
    ]
    skewed_cols = ["triglyceride", "waistline", "HDL_chole", "LDL_chole", "SGOT_AST", "SGOT_ALT"]
    pure_numeric = [col for col in numeric_cols if col not in skewed_cols]
    categorical_cols = ["sex", "SMK_stat_type_cd", "urine_protein", "hear_left", "hear_right"]

    numeric_pipeline = Pipeline(
        steps=[
            ("clip", QuantileClipper()),
            ("scale", RobustScaler()),
        ]
    )

    skew_pipeline = Pipeline(
        steps=[
            ("clip", QuantileClipper()),
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, pure_numeric),
            ("skew", skew_pipeline, skewed_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


MODELS: Dict[str, Tuple[BaseEstimator, Dict[str, List]]] = {
    # Katram modelim glabā estimatoru un tā GridSearch parametrus
    "logreg": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        {"model__C": [0.1, 1.0], "model__solver": ["lbfgs", "liblinear"]},
    ),
    "rf": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 300], "model__max_depth": [15, 25]},
    ),
    "gb": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__learning_rate": [0.05, 0.1], "model__n_estimators": [150, 250]},
    ),
}


def ensure_experiment() -> None:
    # Pārliecinās, ka MLflow eksperiments eksistē un ir aktīvs
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(
    name: str,
    grid: GridSearchCV,
    metrics_dict: Dict[str, float],
    X_train: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> str:
    y_true_bin = (y_true == "Y").astype(int)
    # ROC un CM zīmēšana saglabājas artifacts mapē
    fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="blue", label=f"ROC AUC = {metrics_dict['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    roc_path = artifacts_dir / f"roc_{name}.png"
    plt.savefig(roc_path)
    plt.close()

    cm = confusion_matrix(y_true_bin, (y_pred == "Y").astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["N", "Y"],
        yticklabels=["N", "Y"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    cm_path = artifacts_dir / f"cm_{name}.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    with mlflow.start_run(run_name=f"drk-{name}") as run:
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics_dict)
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(cm_path))
        input_example = X_train.iloc[:2].copy()
        mlflow.sklearn.log_model(
            grid.best_estimator_,
            name="model",
            input_example=input_example,
        )
        mlflow.set_tag("model_name", name)
        return run.info.run_id


def main() -> None:
    # Pilnais treniņa scenārijs: sagatavošana, treniņš, logi un champion reģistrācija
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)
    ensure_experiment()

    run_records = []
    for name, (estimator, param_grid) in MODELS.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, pos_label="Y"),
            "roc_auc": roc_auc_score((y_test == "Y").astype(int), y_proba),
        }
        run_id = log_run(name, grid, metrics_dict, X_train, y_test, y_pred, y_proba)
        run_records.append((name, metrics_dict["accuracy"], run_id))
        print(f"{name} -> {metrics_dict}")

    best_name, _, best_run_id = max(run_records, key=lambda item: item[1])
    print(f"Champion model: {best_name}")
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        registered.version,
    )


if __name__ == "__main__":
    main()
