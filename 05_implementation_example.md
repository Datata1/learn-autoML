# AutoML Implementation Example

Hands-on implementation using the **Iris** (multi-class classification) and **Wine** (multi-class classification) datasets from scikit-learn.

---

## Setup

This project uses `uv` for dependency management. See `pyproject.toml` in the project root.

```bash
# Core dependencies (sklearn, optuna, tpot, ...)
uv sync

# Add PyCaret
uv sync --extra pycaret

# Add Auto-sklearn (Linux/macOS only)
sudo apt install swig build-essential   # Ubuntu/Debian prerequisite
uv sync --extra autosklearn

# Add Optuna visualizations
uv sync --extra viz

# Run this file
uv run python 05_implementation_example.py
```

---

## Part 1: Exploratory Data Analysis (EDA)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine

# ── Iris Dataset ────────────────────────────────────────────────────────────
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["target_name"] = iris_df["target"].map(
    {i: n for i, n in enumerate(iris.target_names)}
)

print("=== Iris Dataset ===")
print(f"Shape: {iris_df.shape}")
print(f"Classes: {iris.target_names}")
print(f"Class distribution:\n{iris_df['target_name'].value_counts()}")
print(f"\nMissing values: {iris_df.isnull().sum().sum()}")
print(iris_df.describe().round(2))

# ── Wine Dataset ─────────────────────────────────────────────────────────────
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["target"] = wine.target
wine_df["target_name"] = wine_df["target"].map(
    {i: n for i, n in enumerate(wine.target_names)}
)

print("\n=== Wine Dataset ===")
print(f"Shape: {wine_df.shape}")
print(f"Classes: {wine.target_names}")
print(f"Class distribution:\n{wine_df['target_name'].value_counts()}")

# ── Correlation Heatmap (Wine) ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(iris_df.drop(["target", "target_name"], axis=1).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Iris: Feature Correlations")

sns.heatmap(wine_df.drop(["target", "target_name"], axis=1).corr(),
            annot=True, fmt=".1f", cmap="coolwarm", ax=axes[1])
axes[1].set_title("Wine: Feature Correlations")

plt.tight_layout()
plt.savefig("eda_correlations.png", dpi=150)
plt.show()
```

---

## Part 2: Baseline — Manual sklearn

Before AutoML, establish a manual baseline to compare against.

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_baselines(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n=== Baseline: {dataset_name} ===")
    print(f"{'Model':<25} {'CV Accuracy':>12} {'Test Accuracy':>14}")
    print("-" * 55)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "test": test_acc}
        print(f"{name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  {test_acc:.4f}")

    return results, X_train, X_test, y_train, y_test

iris_results, iris_X_train, iris_X_test, iris_y_train, iris_y_test = evaluate_baselines(
    iris.data, iris.target, "Iris"
)
wine_results, wine_X_train, wine_X_test, wine_y_train, wine_y_test = evaluate_baselines(
    wine.data, wine.target, "Wine"
)
```

---

## Part 3: HPO with Optuna

Improve the best baseline model using Bayesian HPO.

```python
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Objective: RandomForest on Iris ─────────────────────────────────────────
def rf_objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()

# ── Objective: Multi-model on Wine ───────────────────────────────────────────
def multi_model_objective(trial, X_train, y_train):
    model_name = trial.suggest_categorical("model", ["rf", "gbt", "svm"])

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 300),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            random_state=42, n_jobs=-1
        )
        pipeline = model
    elif model_name == "gbt":
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=42
        )
        pipeline = model
    else:  # SVM
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True) if kernel != "linear" else "scale"
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy").mean()

# ── Run studies ───────────────────────────────────────────────────────────────
print("Running Optuna HPO on Iris (RandomForest)...")
iris_study = optuna.create_study(direction="maximize")
iris_study.optimize(
    lambda trial: rf_objective(trial, iris_X_train, iris_y_train),
    n_trials=100,
    show_progress_bar=True
)

print("\nRunning Optuna HPO on Wine (multi-model)...")
wine_study = optuna.create_study(direction="maximize")
wine_study.optimize(
    lambda trial: multi_model_objective(trial, wine_X_train, wine_y_train),
    n_trials=100,
    show_progress_bar=True
)

# ── Results ──────────────────────────────────────────────────────────────────
print("\n=== Optuna Results ===")
print(f"Iris  - Best CV accuracy: {iris_study.best_value:.4f}")
print(f"Iris  - Best params: {iris_study.best_params}")
print(f"\nWine  - Best CV accuracy: {wine_study.best_value:.4f}")
print(f"Wine  - Best params: {wine_study.best_params}")

# ── Evaluate best model on test set ──────────────────────────────────────────
best_iris_model = RandomForestClassifier(**{
    k: v for k, v in iris_study.best_params.items()
}, random_state=42, n_jobs=-1)
best_iris_model.fit(iris_X_train, iris_y_train)
iris_optuna_score = accuracy_score(iris_y_test, best_iris_model.predict(iris_X_test))
print(f"\nIris  - Optuna model test accuracy: {iris_optuna_score:.4f}")
```

---

## Part 4: TPOT — Genetic Pipeline Search

```python
from tpot import TPOTClassifier

print("Running TPOT on Iris...")
tpot_iris = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    scoring="accuracy",
    verbosity=1,
    random_state=42,
    n_jobs=-1,
    max_time_mins=5       # hard time limit
)
tpot_iris.fit(iris_X_train, iris_y_train)
iris_tpot_score = tpot_iris.score(iris_X_test, iris_y_test)
print(f"TPOT Iris test accuracy: {iris_tpot_score:.4f}")
tpot_iris.export("tpot_iris_pipeline.py")

print("\nRunning TPOT on Wine...")
tpot_wine = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    scoring="accuracy",
    verbosity=1,
    random_state=42,
    n_jobs=-1,
    max_time_mins=5
)
tpot_wine.fit(wine_X_train, wine_y_train)
wine_tpot_score = tpot_wine.score(wine_X_test, wine_y_test)
print(f"TPOT Wine test accuracy: {wine_tpot_score:.4f}")
tpot_wine.export("tpot_wine_pipeline.py")
```

---

## Part 5: PyCaret — Low-Code AutoML

```python
from pycaret.classification import (
    setup, compare_models, tune_model, blend_models,
    finalize_model, predict_model, pull, evaluate_model
)
import pandas as pd

def run_pycaret(X_train, X_test, y_train, y_test, target_names, dataset_name):
    print(f"\n=== PyCaret: {dataset_name} ===")

    train_df = pd.DataFrame(X_train)
    train_df.columns = [f"f{i}" for i in range(X_train.shape[1])]
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test)
    test_df.columns = [f"f{i}" for i in range(X_test.shape[1])]
    test_df["target"] = y_test

    exp = setup(
        data=train_df,
        target="target",
        session_id=42,
        verbose=False,
        html=False
    )

    top3 = compare_models(n_select=3, sort="Accuracy", verbose=False)
    results = pull()
    print(results[["Model", "Accuracy", "Recall", "Prec.", "F1"]].head(5).to_string())

    tuned = tune_model(top3[0], n_iter=30, verbose=False)
    blended = blend_models(top3, verbose=False)

    final = finalize_model(blended)
    preds = predict_model(final, data=test_df.drop("target", axis=1))

    test_accuracy = accuracy_score(y_test, preds["prediction_label"])
    print(f"\nBlended ensemble test accuracy: {test_accuracy:.4f}")
    return test_accuracy

pycaret_iris_score = run_pycaret(
    iris_X_train, iris_X_test, iris_y_train, iris_y_test,
    iris.target_names, "Iris"
)
pycaret_wine_score = run_pycaret(
    wine_X_train, wine_X_test, wine_y_train, wine_y_test,
    wine.target_names, "Wine"
)
```

---

## Part 6: Auto-sklearn (Linux/macOS only)

```python
# Only run this on Linux or macOS with swig installed
import autosklearn.classification

def run_autosklearn(X_train, X_test, y_train, y_test, dataset_name):
    print(f"\n=== Auto-sklearn: {dataset_name} ===")
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        memory_limit=3072,
        n_jobs=-1,
        seed=42
    )
    automl.fit(X_train, y_train)

    test_score = automl.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    print("Leaderboard:")
    print(automl.leaderboard())
    print("Sprint statistics:", automl.sprint_statistics())
    return test_score

# autosklearn_iris = run_autosklearn(iris_X_train, iris_X_test, iris_y_train, iris_y_test, "Iris")
# autosklearn_wine = run_autosklearn(wine_X_train, wine_X_test, wine_y_train, wine_y_test, "Wine")
```

---

## Part 7: Results Comparison & Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# ── Compile results ───────────────────────────────────────────────────────────
iris_comparison = {
    "Logistic Regression": iris_results["Logistic Regression"]["test"],
    "Random Forest (default)": iris_results["Random Forest"]["test"],
    "Gradient Boosting (default)": iris_results["Gradient Boosting"]["test"],
    "SVM RBF (default)": iris_results["SVM (RBF)"]["test"],
    "Optuna (RF, 100 trials)": iris_optuna_score,
    "TPOT (5 gen × 20 pop)": iris_tpot_score,
    "PyCaret (blended ensemble)": pycaret_iris_score,
}

wine_comparison = {
    "Logistic Regression": wine_results["Logistic Regression"]["test"],
    "Random Forest (default)": wine_results["Random Forest"]["test"],
    "Gradient Boosting (default)": wine_results["Gradient Boosting"]["test"],
    "SVM RBF (default)": wine_results["SVM (RBF)"]["test"],
    "Optuna (multi-model, 100 trials)": wine_study.best_value,  # CV score
    "TPOT (5 gen × 20 pop)": wine_tpot_score,
    "PyCaret (blended ensemble)": pycaret_wine_score,
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ["#4c72b0"] * 4 + ["#dd8452", "#55a868", "#c44e52"]

for ax, comparison, title in zip(axes,
    [iris_comparison, wine_comparison],
    ["Iris Dataset Results", "Wine Dataset Results"]):

    models = list(comparison.keys())
    scores = list(comparison.values())
    bars = ax.barh(models, scores, color=colors)
    ax.set_xlim(0.7, 1.02)
    ax.set_xlabel("Test Accuracy")
    ax.set_title(title)
    ax.axvline(x=max(scores[:4]), color="gray", linestyle="--", alpha=0.5, label="Best baseline")

    for bar, score in zip(bars, scores):
        ax.text(score + 0.002, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", fontsize=9)

    ax.legend()

plt.tight_layout()
plt.savefig("automl_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Optuna visualization ──────────────────────────────────────────────────────
optuna.visualization.plot_optimization_history(wine_study).write_image("optuna_history.png")
optuna.visualization.plot_param_importances(wine_study).write_image("optuna_importances.png")
```

---

## Part 8: Deep Dive — Model Evaluation

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

# ── Confusion Matrix ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Best Optuna model on Iris
iris_preds = best_iris_model.predict(iris_X_test)
ConfusionMatrixDisplay(
    confusion_matrix(iris_y_test, iris_preds),
    display_labels=iris.target_names
).plot(ax=axes[0], cmap="Blues")
axes[0].set_title(f"Iris (Optuna RF) — Acc: {iris_optuna_score:.4f}")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# ── Full classification report ────────────────────────────────────────────────
print("=== Iris Classification Report (Optuna RF) ===")
print(classification_report(iris_y_test, iris_preds, target_names=iris.target_names))
```

---

## Summary of Results

| Method | Iris Test Acc | Wine Test Acc | Notes |
|--------|--------------|--------------|-------|
| Best baseline (manual) | ~0.97 | ~0.97 | Gradient Boosting |
| Optuna HPO (100 trials) | ~0.97–1.00 | ~0.97–1.00 | Bayesian optimization |
| TPOT (5 gen) | ~0.97 | ~0.97 | Genetic programming |
| PyCaret (blend) | ~0.97–1.00 | ~0.97–1.00 | Ensemble voting |
| Auto-sklearn (120s) | ~1.00 | ~0.99–1.00 | Meta-learning + Bayesian |

> **Key insight**: Both Iris and Wine are relatively easy datasets where most classifiers achieve >95%. The real value of AutoML shows on harder, messier, real-world datasets.

---

## Next Steps

1. Try with a harder dataset: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or [Adult Income](https://archive.ics.uci.edu/dataset/2/adult)
2. Add SHAP analysis for explainability
3. Benchmark runtimes: how long did each framework take?
4. Try with imbalanced classes (add `class_weight="balanced"`)
5. Export and deploy the best model with `joblib.dump(model, "model.pkl")`
