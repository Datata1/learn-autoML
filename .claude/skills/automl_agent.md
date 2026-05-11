# Skill: AutoML Implementation Agent

This skill enables Claude Code agents to implement AutoML experiments on tabular datasets. Use it when asked to run AutoML, tune models, or compare ML frameworks.

---

## Knowledge Base Location

All reference material is in `/home/jan-david/Documents/automl/`:

| File | Content | Read when... |
|------|---------|-------------|
| `01_foundations.md` | What AutoML is, CASH problem, meta-learning | Explaining AutoML concepts |
| `02_pipeline_components.md` | Preprocessing, feature engineering, ensembling | Building or debugging a pipeline |
| `03_hyperparameter_optimization.md` | Grid/Random/Bayesian/Hyperband with code | Choosing or implementing HPO |
| `04_frameworks_comparison.md` | Auto-sklearn, TPOT, PyCaret, H2O, Optuna | Recommending or using a framework |
| `05_implementation_example.md` | Full working code for Iris & Wine | Implementing AutoML from scratch |

---

## Decision Tree: Which Framework to Use?

```
1. Is the task classification or regression on tabular data?
   └── Yes → continue
   └── No (images/text) → use AutoKeras

2. Is the OS Linux or macOS?
   └── Yes → Auto-sklearn is available (best search quality)
   └── No (Windows) → use PyCaret, TPOT, or Optuna

3. What is the priority?
   ├── Fastest to results          → PyCaret
   ├── Best accuracy               → Auto-sklearn or Optuna
   ├── Interpretable pipeline      → TPOT (exports Python code)
   ├── Large dataset / production  → H2O AutoML
   └── Custom model, own code      → Optuna
```

---

## Standard Implementation Template

### Step 1: Data preparation
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify for classification
)
```

### Step 2: Establish baseline
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
baseline = RandomForestClassifier(n_estimators=100, random_state=42)
cv_score = cross_val_score(baseline, X_train, y_train, cv=5, scoring="accuracy").mean()
print(f"Baseline CV: {cv_score:.4f}")
```

### Step 3: HPO with Optuna (always available, no OS restriction)
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(model, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print("Best:", study.best_params, study.best_value)
```

### Step 4: Evaluate
```python
from sklearn.metrics import accuracy_score, classification_report
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
test_score = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test accuracy: {test_score:.4f}")
print(classification_report(y_test, best_model.predict(X_test)))
```

---

## Checklist for AutoML Experiments

- [ ] EDA: check shape, missing values, class distribution
- [ ] Split data stratified (classification) or random (regression)
- [ ] Establish a manual baseline (at least 2-3 algorithms)
- [ ] Run HPO with budget (time limit or n_trials)
- [ ] Evaluate on held-out test set (not the CV set)
- [ ] Compare: does AutoML beat the baseline?
- [ ] Analyze: what hyperparameters matter most? (Optuna: `plot_param_importances`)
- [ ] Document: runtime, best params, train/test accuracy

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| HPO on test set → data leakage | Always tune on cross-validation, evaluate on a held-out test set |
| Too few trials with Bayesian Opt | Use at least 50–100 trials for meaningful results |
| Auto-sklearn on Windows | Use PyCaret or Optuna instead |
| Forgetting to scale for SVM/LR | Wrap in `Pipeline([("scaler", StandardScaler()), ("clf", model)])` |
| TPOT runs forever | Set `max_time_mins=5` and `generations=5` for quick experiments |

---

## Metrics Reference

| Task | Primary metric | Secondary |
|------|---------------|-----------|
| Binary classification | roc_auc | f1, accuracy |
| Multi-class classification | accuracy | f1_weighted, log_loss |
| Regression | r2 | rmse, mae |
| Imbalanced classification | f1_weighted | roc_auc, precision_recall_auc |

---

## Agent Instructions

When implementing AutoML for a user:

1. **Ask** for dataset path/name and task type (classification/regression) if not specified
2. **Read** `05_implementation_example.md` for complete working code patterns
3. **Read** `04_frameworks_comparison.md` if unsure which framework to use
4. **Always** establish a baseline before AutoML
5. **Always** report both CV score (tuning) and test score (final evaluation) separately
6. **Prefer Optuna** for portability; use Auto-sklearn only when on Linux/macOS
7. **Export** the best model with `joblib.dump(model, "best_model.pkl")`
