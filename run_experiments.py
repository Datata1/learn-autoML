"""
AutoML Experiment Runner
========================
Runs all experiments and saves results to results/ for the Marimo notebooks.

Usage:
    uv run python run_experiments.py
    uv run python run_experiments.py --no-tpot   # skip TPOT (faster)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Data ────────────────────────────────────────────────────────────────────

def load_datasets() -> dict:
    splits = {}
    for name, loader in [("iris", load_iris), ("wine", load_wine)]:
        raw = loader()
        X_tr, X_te, y_tr, y_te = train_test_split(
            raw.data, raw.target, test_size=0.2, random_state=42, stratify=raw.target
        )
        splits[name] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "feature_names": list(raw.feature_names),
            "target_names": list(raw.target_names),
            "n_samples": len(raw.data),
            "n_features": raw.data.shape[1],
            "n_classes": len(raw.target_names),
        }
    return splits


# ── Baselines ────────────────────────────────────────────────────────────────

def run_baselines(X_train, X_test, y_train, y_test) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ]),
    }
    results = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="accuracy")
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        elapsed = round(time.perf_counter() - t0, 2)
        results[name] = {
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "test": round(float(test_acc), 4),
            "time_s": elapsed,
        }
        print(f"  {name:<26} CV {cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
              f"Test {test_acc:.4f}  ({elapsed:.1f}s)")
    return results


# ── Optuna HPO ───────────────────────────────────────────────────────────────

def _build_model(trial):
    """Conditional multi-model search space."""
    algo = trial.suggest_categorical("algo", ["rf", "gbt", "svm"])
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 400),
            max_depth=trial.suggest_int("max_depth", 2, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            random_state=42, n_jobs=-1,
        )
    if algo == "gbt":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=42,
        )
    # svm
    C = trial.suggest_float("C", 1e-2, 100.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)),
    ])


def run_optuna(X_train, X_test, y_train, y_test, study_name: str, n_trials: int = 100):
    storage = f"sqlite:///{RESULTS_DIR}/optuna.db"

    def objective(trial):
        model = _build_model(trial)
        return cross_val_score(model, X_train, y_train, cv=CV, scoring="accuracy").mean()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = round(time.perf_counter() - t0, 2)

    # Rebuild and evaluate best model on test set
    best_trial = study.best_trial
    best_model = _build_model(best_trial)
    best_model.fit(X_train, y_train)
    test_acc = float(accuracy_score(y_test, best_model.predict(X_test)))

    joblib.dump(best_model, RESULTS_DIR / f"{study_name}_best_model.pkl")

    return {
        "cv_score": round(float(study.best_value), 4),
        "test": round(test_acc, 4),
        "best_params": study.best_params,
        "n_trials": n_trials,
        "time_s": elapsed,
    }


# ── TPOT ─────────────────────────────────────────────────────────────────────

def run_tpot(X_train, X_test, y_train, y_test, dataset_name: str):
    try:
        from tpot import TPOTClassifier
    except ImportError:
        print("  TPOT not installed — skipping (uv sync --extra tpot)")
        return None

    t0 = time.perf_counter()
    tpot = TPOTClassifier(
        generations=5, population_size=20,
        cv=5, scoring="accuracy",
        verbosity=1, random_state=42,
        n_jobs=-1, max_time_mins=4,
    )
    tpot.fit(X_train, y_train)
    test_acc = float(tpot.score(X_test, y_test))
    elapsed = round(time.perf_counter() - t0, 2)
    tpot.export(str(RESULTS_DIR / f"tpot_{dataset_name}_pipeline.py"))
    return {"test": round(test_acc, 4), "time_s": elapsed}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(run_tpot_flag: bool = True):
    datasets = load_datasets()
    all_results: dict = {}

    for name, data in datasets.items():
        X_tr, X_te = data["X_train"], data["X_test"]
        y_tr, y_te = data["y_train"], data["y_test"]

        print(f"\n{'='*60}")
        print(f"  {name.upper()}  ({data['n_samples']} samples, "
              f"{data['n_features']} features, {data['n_classes']} classes)")
        print(f"{'='*60}")

        print("\n[1/3] Baselines")
        baselines = run_baselines(X_tr, X_te, y_tr, y_te)

        print(f"\n[2/3] Optuna (100 trials, multi-model search space)")
        optuna_res = run_optuna(X_tr, X_te, y_tr, y_te, study_name=f"{name}_automl")
        print(f"  Best CV {optuna_res['cv_score']:.4f}  "
              f"Test {optuna_res['test']:.4f}  "
              f"({optuna_res['time_s']:.1f}s)")
        print(f"  Best config: {optuna_res['best_params']}")

        tpot_res = None
        if run_tpot_flag:
            print(f"\n[3/3] TPOT (5 generations × 20 population)")
            tpot_res = run_tpot(X_tr, X_te, y_tr, y_te, name)
            if tpot_res:
                print(f"  Test {tpot_res['test']:.4f}  ({tpot_res['time_s']:.1f}s)")

        all_results[name] = {
            "baselines": baselines,
            "optuna": optuna_res,
            "tpot": tpot_res,
            "meta": {
                "n_samples": data["n_samples"],
                "n_features": data["n_features"],
                "n_classes": data["n_classes"],
                "feature_names": data["feature_names"],
                "target_names": data["target_names"],
            },
        }

    out = RESULTS_DIR / "experiments.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out}")
    print("\nOpen the notebooks with:")
    print("  uv run marimo edit notebooks/01_eda.py")
    print("  uv run marimo edit notebooks/02_results.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tpot", action="store_true", help="Skip TPOT")
    args = parser.parse_args()
    main(run_tpot_flag=not args.no_tpot)
