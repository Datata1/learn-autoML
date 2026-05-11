# Hyperparameter Optimization (HPO)

HPO is the engine of AutoML. This document covers every major method, their trade-offs, and when to use each.

---

## What are Hyperparameters?

**Parameters** are learned from data during training (e.g., weights in a neural network, split thresholds in a tree).

**Hyperparameters** are set *before* training and control the learning process:

| Model | Example Hyperparameters |
|-------|------------------------|
| Random Forest | n_estimators, max_depth, min_samples_split |
| SVM | C, kernel, gamma |
| Neural Network | learning_rate, batch_size, num_layers, dropout |
| Gradient Boosting | n_estimators, learning_rate, max_depth, subsample |
| Ridge Regression | alpha (regularization strength) |

---

## 1. Grid Search

### How it works
Define a discrete grid over all hyperparameter combinations. Try every combination.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, grid_search.best_score_)
```

**Total evaluations**: 3 × 3 × 2 = **18 combinations × 5 folds = 90 model fits**

### Pros & Cons
| Pros | Cons |
|------|------|
| Exhaustive, guaranteed to find best in grid | Exponential scaling (curse of dimensionality) |
| Fully parallelizable | Wastes budget on unimportant hyperparameters |
| Simple to understand and implement | Can miss good values between grid points |

### When to use
- Very few hyperparameters (≤ 3)
- Small search space
- Cheap model training

---

## 2. Random Search

### How it works
Sample hyperparameter configurations **randomly** from defined distributions. Each configuration is independent.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(10, 500),
    "max_depth": randint(1, 50),
    "min_samples_split": randint(2, 20),
    "max_features": uniform(0.1, 0.9)  # fraction
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,        # number of configurations to try
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

### Why Random Search beats Grid Search

Key insight (Bergstra & Bengio, 2012): In any search space, most hyperparameters are **unimportant**. Only a few matter (often 1–3 out of 10+).

```
Grid Search with 3 values per hyperparameter, 9 hyperparameters:
→ 3^9 = 19,683 evaluations

But if only 2 hyperparameters matter:
- Grid Search: still tries 19,683 combinations, wastes 3^7 = 2,187 per value
- Random Search: 100 evaluations, each independently covers the important dimensions
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Scales well to many hyperparameters | No exploitation of previous results |
| Efficient when few params matter | Random, so results vary between runs |
| Easy to add more iterations | Can miss narrow optimal regions |
| Parallelizable | |

### When to use
- Many hyperparameters (> 5)
- Unclear which parameters matter
- Quick baseline before applying Bayesian Opt

---

## 3. Bayesian Optimization

### Core Idea
Build a **probabilistic surrogate model** of the objective function (validation loss vs. hyperparameters). Use it to intelligently choose the next configuration to evaluate.

```
Iteration 1: Try config x₁ → observe f(x₁)
Iteration 2: Try config x₂ → observe f(x₂)
...
After n iterations: fit surrogate model p(f | x₁...xₙ)
Next config: maximize acquisition function A(x) using surrogate
```

### Surrogate Models

**Gaussian Process (GP)**:
- Assumes the objective is a smooth function
- Provides mean + uncertainty estimate for any point
- Expensive for many observations (O(n³))
- Best for: low-dimensional, continuous search spaces

**Tree-structured Parzen Estimator (TPE)**:
- Models p(x | good) and p(x | bad) separately using KDEs
- Acquisition: maximize p(x | good) / p(x | bad)
- Scales well to high dimensions and categorical variables
- Used by: **Hyperopt**, **Optuna** (default)

**Random Forest Surrogate (SMAC)**:
- Uses a Random Forest to model the objective
- Handles conditional and categorical hyperparameters natively
- Used by: **Auto-sklearn** (SMAC3)

### Acquisition Functions

| Function | Formula | Behaviour |
|----------|---------|-----------|
| Expected Improvement (EI) | E[max(f(x) - f*, 0)] | Balances exploration/exploitation |
| Upper Confidence Bound (UCB) | μ(x) + κσ(x) | κ controls exploration |
| Probability of Improvement (PI) | P(f(x) > f*) | More exploitative |

### Implementation with Optuna (TPE)

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth = trial.suggest_int("max_depth", 2, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    max_features = trial.suggest_float("max_features", 0.1, 1.0)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best params:", study.best_params)
print("Best score:", study.best_value)
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Uses previous evaluations intelligently | Harder to parallelize (sequential by design) |
| Often finds good configs in fewer evaluations | Surrogate model has its own hyperparameters |
| Well-studied, strong theoretical guarantees | Slow for very cheap objective functions |

---

## 4. Hyperband & ASHA

### The Multi-Fidelity Idea

Key insight: **Most bad configurations are bad early**. You can evaluate them with fewer resources (fewer epochs, smaller data subset) and discard them before investing full training time.

### Successive Halving (SHA)

```
Start:    n configurations, budget b each
Round 1:  train all n for b epochs → keep top 1/η
Round 2:  train remaining for η*b epochs → keep top 1/η
...
Until:    1 configuration remains, trained for full budget
```

### Hyperband

Hyperband addresses SHA's weakness: you don't know the optimal `n` (many cheap vs. few expensive).

Hyperband runs multiple SHA brackets with different starting configurations:
- **More brackets** = try more diverse configurations (more random search)
- **Fewer brackets** = exploit longer (more resources per config)

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=100,  # e.g., number of epochs
        reduction_factor=3
    )
)
```

### ASHA (Asynchronous Successive Halving)

ASHA is the **parallel** version of SHA:
- Workers don't wait for each other
- Promotion happens asynchronously
- Much better GPU/CPU utilization
- Used by: **Ray Tune**, **Optuna** (async mode)

### Comparison: Grid vs Random vs Bayesian vs Hyperband

| Method | Parallelizable | Uses history | Early stopping | Best for |
|--------|---------------|-------------|----------------|----------|
| Grid Search | Yes | No | No | Small discrete spaces |
| Random Search | Yes | No | No | Many hyperparameters |
| Bayesian Opt | Partial | Yes | No | Few expensive evaluations |
| Hyperband / ASHA | Yes (ASHA) | No | Yes | Neural networks, expensive training |
| BOHB | Yes | Yes | Yes | Best of Bayesian + Hyperband |

**BOHB** (Bayesian Optimization + HyperBand) is often the strongest method for deep learning HPO.

---

## 5. Population-Based Training (PBT)

### How it works

PBT trains a **population** of models simultaneously and uses evolutionary mechanisms:

1. Initialize population of N models with random hyperparameters
2. Train all models for some steps
3. **Exploit**: copy weights from better-performing models to worse ones
4. **Explore**: perturb the hyperparameters of the copied model
5. Repeat

Unlike standard HPO, PBT can **change hyperparameters during training** (e.g., gradually increase batch size, decay learning rate).

```
Model 1: lr=0.1  → performance 0.85 → EXPLOIT (copy from Model 3)
Model 2: lr=0.01 → performance 0.72 → EXPLOIT
Model 3: lr=0.05 → performance 0.91 → keep, EXPLORE (lr → 0.048)
```

### When to use
- Deep learning, reinforcement learning
- When you want to schedule hyperparameters (annealing)
- Google and DeepMind use this extensively

---

## 6. Neural Architecture Search (NAS)

NAS automates the design of neural network architectures — the structure, not just hyperparameters.

### Search Space Components
- Number and type of layers (Conv, LSTM, Attention, Dense)
- Layer dimensions (filters, units)
- Activation functions
- Skip connections (ResNet-style)
- Normalization (BatchNorm, LayerNorm)

### NAS Strategies

**Evolutionary / RL-based (original NAS, Zoph & Le 2017)**:
- Use a controller RNN to generate architectures
- Train child network, get validation accuracy
- Use as reward to update controller
- Very expensive (800 GPUs for 28 days in original paper)

**Differentiable Architecture Search (DARTS, 2018)**:
- Relax discrete architecture choices to continuous weights
- Use gradient descent to optimize architecture weights jointly with model weights
- ~1000x faster than RL-based NAS
- Used in: AutoKeras, many production NAS systems

**One-shot / Weight Sharing (ENAS, 2018)**:
- Train a single supernetwork containing all candidate architectures
- Sub-architectures share weights
- Evaluate candidates by sampling from the supernetwork
- Much cheaper: weights trained once, architectures evaluated in seconds

```python
import autokeras as ak

# AutoKeras uses NAS under the hood
clf = ak.StructuredDataClassifier(max_trials=10, overwrite=True)
clf.fit(X_train, y_train, epochs=20)
print("Test accuracy:", clf.evaluate(X_test, y_test))
```

---

## HPO Method Selection Guide

```
Is your training fast (< 1 min per run)?
├── Yes, few hyperparameters (≤ 3):     Grid Search
├── Yes, many hyperparameters (> 3):    Random Search
└── No (slow training):
    ├── Have GPU/distributed compute?   ASHA / Hyperband
    ├── Sequential evaluations OK?      Bayesian Optimization (TPE)
    └── Deep learning + want scheduling? Population-Based Training
```
