# AutoML Frameworks Comparison

A practical guide to the major AutoML frameworks — what they are, how they work, and when to choose each.

---

## Framework Overview

| Framework | Type | Backend | HPO Method | Best for |
|-----------|------|---------|------------|----------|
| Auto-sklearn | Full pipeline | scikit-learn | Bayesian (SMAC) + meta-learning | Research, strong baselines |
| TPOT | Full pipeline | scikit-learn | Genetic programming | Pipeline diversity, interpretable outputs |
| PyCaret | Full pipeline | sklearn/XGB/LGB | Multiple (incl. random) | Quick EDA + modeling, presentations |
| H2O AutoML | Full pipeline | H2O (Java) | Random + Stacked Ensembles | Production, large data, Spark |
| AutoKeras | NAS (deep learning) | Keras/TF | NAS (Bayesian + DARTS) | Unstructured data (images, text, tabular) |
| Optuna | HPO only | Any (custom) | TPE + Hyperband | Custom models, flexible HPO |
| Ray Tune | HPO only | Any (custom) | ASHA, PBT, BOHB | Distributed training, large scale |
| FLAML | Full pipeline | sklearn + LGB | Cost-aware search | Resource-constrained environments |

---

## 1. Auto-sklearn

**GitHub**: `automl/auto-sklearn` | **Install**: `pip install auto-sklearn`

### Architecture
Auto-sklearn wraps scikit-learn with a Bayesian Optimization loop using **SMAC** (Sequential Model-based Algorithm Configuration) and initializes with **meta-learning** from 140+ previous datasets.

```
                  Meta-Features
                       │
         ┌─────────────▼──────────────┐
         │  Meta-learning Warm Start  │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │    SMAC (Bayesian Opt)     │
         │  ┌─────────────────────┐  │
         │  │  Pipeline space:    │  │
         │  │  - 15 classifiers   │  │
         │  │  - 14 preprocessors │  │
         │  └─────────────────────┘  │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   Greedy Ensemble Builder  │
         └────────────────────────────┘
```

### Key Features
- Automatic warm-starting from meta-database
- Ensemble selection from all evaluated models
- Supports classification and regression
- Time and memory budgets
- Feature preprocessing: PCA, polynomial, one-hot, etc.

### Basic Usage
```python
import autosklearn.classification
import autosklearn.regression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,      # total search time in seconds
    per_run_time_limit=30,            # max time per individual model
    memory_limit=3072,                # MB
    n_jobs=-1
)
automl.fit(X_train, y_train)
print("Score:", automl.score(X_test, y_test))
print("Models:", automl.leaderboard())
```

### Limitations
- Linux/macOS only (uses `pynisher` for memory limiting)
- Requires `swig` for compilation
- Slower than PyCaret for quick exploration

---

## 2. TPOT

**GitHub**: `EpistasisLab/tpot` | **Install**: `pip install tpot`

### Architecture
TPOT uses **genetic programming** to evolve entire ML pipelines as tree structures.

```
Pipeline as tree:
                    [output: predictions]
                           │
                    [RandomForest]
                    /            \
           [PCA]              [SelectKBest]
             │                      │
        [StandardScaler]      [PolynomialFeatures]
             │                      │
           Input                  Input
```

TPOT's genetic algorithm:
1. Initialize population of random pipelines
2. Evaluate each pipeline via cross-validation
3. Select best pipelines (tournament selection)
4. Apply mutation (change a step) and crossover (combine two pipelines)
5. Repeat for N generations

### Key Features
- Outputs a **Python script** with the best pipeline → fully interpretable
- Searches for entire pipeline structure, not just hyperparameters
- Parallelizable with `n_jobs`

### Basic Usage
```python
from tpot import TPOTClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

tpot = TPOTClassifier(
    generations=5,         # number of generations
    population_size=20,    # pipelines per generation
    verbosity=2,
    random_state=42,
    n_jobs=-1
)
tpot.fit(X_train, y_train)
print("Score:", tpot.score(X_test, y_test))

# Export the best pipeline as Python code
tpot.export("best_pipeline.py")
```

### Exported pipeline example
```python
# Exported by TPOT
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=100
    )
)
exported_pipeline.fit(X_train, y_train)
```

---

## 3. PyCaret

**GitHub**: `pycaret/pycaret` | **Install**: `pip install pycaret[full]`

### Architecture
PyCaret is a **low-code ML library** that wraps 20+ algorithms with a unified API. Less focused on automated search, more on streamlining the ML workflow.

```
setup() → compare_models() → tune_model() → blend_models() → stack_models() → finalize_model() → predict_model()
```

### Key Features
- Extremely quick to get started
- Automatic preprocessing in `setup()`
- `compare_models()` ranks all algorithms with cross-validation
- Experiment logging (MLflow integration)
- Explainability plots (SHAP, feature importance)

### Basic Usage
```python
from pycaret.classification import *
from sklearn.datasets import load_iris
import pandas as pd

# Prepare data as DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Setup: defines target, handles preprocessing automatically
exp = setup(
    data=df,
    target="target",
    session_id=42,
    verbose=False
)

# Compare all models
best_models = compare_models(n_select=3, sort="Accuracy")

# Tune the best model
tuned_model = tune_model(best_models[0], n_iter=50)

# Ensemble
blended = blend_models(best_models)

# Evaluate
evaluate_model(tuned_model)

# Finalize and predict
final_model = finalize_model(tuned_model)
predictions = predict_model(final_model, data=df.drop("target", axis=1))
```

### Limitations
- Less sophisticated HPO than Auto-sklearn/Optuna
- Not ideal for very large datasets
- More of a workflow tool than a true AutoML optimizer

---

## 4. H2O AutoML

**GitHub**: `h2oai/h2o-3` | **Install**: `pip install h2o`

### Architecture
H2O is a **Java-based distributed ML platform**. AutoML runs on top of H2O's core algorithms and produces a **stacked ensemble** as the final model.

```
Training order:
1. XGBoost variants (multiple configs)
2. GBM variants
3. Deep Learning
4. Random Forest + Extremely Randomized Trees
5. GLM (Generalized Linear Model)
6. Stacked Ensemble (all models)
7. Stacked Ensemble (best of each family)
```

### Key Features
- Scales to very large datasets (distributed, Spark-compatible)
- Production-ready (model export to MOJO/POJO for Java deployment)
- AutoML Leaderboard — ranked comparison of all models
- Native handling of imbalanced classes

### Basic Usage
```python
import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_wine
import pandas as pd

h2o.init()

# Load data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target.astype(str)   # H2O needs string for classification
h2o_df = h2o.H2OFrame(df)

train, test = h2o_df.split_frame(ratios=[0.8], seed=42)

# Run AutoML
aml = H2OAutoML(max_runtime_secs=120, seed=42)
aml.train(y="target", training_frame=train)

# Leaderboard
print(aml.leaderboard)

# Predict
preds = aml.leader.predict(test)
performance = aml.leader.model_performance(test)
print("Accuracy:", performance)

h2o.shutdown()
```

---

## 5. AutoKeras

**GitHub**: `keras-team/autokeras` | **Install**: `pip install autokeras`

### Architecture
AutoKeras applies **Neural Architecture Search** using a **Bayesian Optimization over a graph of Keras blocks**.

Supports:
- `ImageClassifier` / `ImageRegressor`
- `TextClassifier` / `TextRegressor`
- `StructuredDataClassifier` / `StructuredDataRegressor`
- Custom pipelines combining blocks

### Basic Usage (Structured/Tabular Data)
```python
import autokeras as ak
import numpy as np

clf = ak.StructuredDataClassifier(
    max_trials=10,       # number of different architectures to try
    overwrite=True
)
clf.fit(X_train, y_train, epochs=50)

accuracy = clf.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Export as Keras model
model = clf.export_model()
model.summary()
```

### When to use AutoKeras
- You want neural networks for tabular/text/image data
- You have GPU available
- You want to compare DL architectures automatically

---

## 6. Optuna

**GitHub**: `optuna/optuna` | **Install**: `pip install optuna`

### Architecture
Optuna is a **framework-agnostic HPO library** — you define an `objective` function and Optuna optimizes it. Not a full AutoML system, but often the best tool when you already know your model family.

### Key Features
- TPE sampler (default, excellent performance)
- Pruning (Hyperband, Median, Percentile)
- Parallelization via `study.optimize(n_jobs=...)`
- Visualization dashboard
- Multi-objective optimization
- Integration with sklearn, PyTorch, TensorFlow, XGBoost, LightGBM

### Advanced Usage
```python
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)

    if kernel in ["rbf", "poly"]:
        gamma = trial.suggest_float("gamma", 1e-5, 10, log=True)
    else:
        gamma = "scale"

    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3

    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    return cross_val_score(model, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
print("Best CV score:", study.best_value)

# Visualize
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

---

## Framework Selection Guide

```
What's your primary goal?
│
├── Quick baseline, minimal code      → PyCaret
│
├── Best accuracy, research setting   → Auto-sklearn + Optuna
│
├── Understand the final pipeline     → TPOT (exports Python code)
│
├── Production / large-scale data     → H2O AutoML
│
├── Deep learning (images, text)      → AutoKeras
│
├── Custom model, flexible HPO        → Optuna or Ray Tune
│
└── Distributed / GPU cluster         → Ray Tune + ASHA/BOHB
```

---

## Quantitative Comparison

| Framework | Setup complexity | Search quality | Training speed | Deployment | Windows support |
|-----------|-----------------|----------------|----------------|------------|-----------------|
| Auto-sklearn | Medium | ⭐⭐⭐⭐⭐ | Slow | Poor | No |
| TPOT | Low | ⭐⭐⭐⭐ | Slow | Good (exports code) | Yes |
| PyCaret | Very low | ⭐⭐⭐ | Fast | Medium | Yes |
| H2O AutoML | Medium | ⭐⭐⭐⭐ | Fast | Excellent (MOJO) | Yes |
| AutoKeras | Medium | ⭐⭐⭐⭐ (DL) | Needs GPU | Good (Keras) | Yes |
| Optuna | High (custom) | ⭐⭐⭐⭐⭐ | Depends | Depends | Yes |
