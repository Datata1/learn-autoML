# AutoML Pipeline Components

An AutoML pipeline mirrors a manual ML pipeline but automates the search over each stage. This document covers every component in depth.

---

## Pipeline Overview

```
Raw Data
    │
    ▼
[1] Data Preprocessing
    ├── Missing value imputation
    ├── Outlier handling
    ├── Type detection & casting
    └── Train/validation/test split
    │
    ▼
[2] Feature Engineering
    ├── Feature selection
    ├── Feature transformation
    ├── Feature generation
    └── Dimensionality reduction
    │
    ▼
[3] Algorithm Selection
    └── Choose from search space of models
    │
    ▼
[4] Hyperparameter Optimization
    └── Tune the selected algorithm(s)
    │
    ▼
[5] Ensembling
    └── Combine top models
    │
    ▼
Final Model / Predictions
```

---

## 1. Automated Data Preprocessing

### Missing Value Imputation
AutoML systems try multiple strategies:

| Strategy | When to use |
|----------|-------------|
| Mean/Median imputation | Numerical, low missingness |
| Mode imputation | Categorical |
| K-NN imputation | Correlated features, small datasets |
| Iterative imputation (MICE) | Complex patterns, higher missingness |
| Indicator column (+ imputation) | When missingness itself is informative |

### Categorical Encoding
| Encoding | Description | Best for |
|----------|-------------|----------|
| One-Hot Encoding | Binary column per category | Low cardinality (< 15 unique) |
| Label Encoding | Integer per category | Ordinal data |
| Target Encoding | Mean of target per category | High cardinality, tree models |
| Frequency Encoding | Count of occurrences | High cardinality |
| Hashing | Hash into fixed bucket size | Very high cardinality |

### Feature Scaling
| Method | Formula | Sensitive to outliers |
|--------|---------|----------------------|
| StandardScaler | (x - mean) / std | Yes |
| MinMaxScaler | (x - min) / (max - min) | Yes |
| RobustScaler | (x - median) / IQR | No |
| Normalizer (L2) | x / ||x|| | No |

> **Note**: Tree-based models (RandomForest, XGBoost) do NOT need scaling. Linear models and neural networks do.

---

## 2. Automated Feature Engineering

### Feature Selection
Reduces dimensionality and removes noise:

| Method | Type | Description |
|--------|------|-------------|
| Variance Threshold | Filter | Remove near-zero variance features |
| Correlation Threshold | Filter | Remove highly correlated features |
| SelectKBest (chi2, ANOVA) | Filter | Statistical tests |
| Recursive Feature Elimination | Wrapper | Iteratively remove weakest features |
| Lasso / ElasticNet | Embedded | Regularization zeroes weak features |
| Feature Importance (tree) | Embedded | Use model's built-in feature scores |

### Feature Transformation
- **Polynomial features**: create `x1*x2`, `x1²`, `x2²` — useful for linear models
- **Log/sqrt transform**: reduce skewness in numerical features
- **Binning**: convert continuous to ordinal
- **Date decomposition**: extract year, month, day-of-week, hour from timestamps

### Automated Feature Generation (Deep Feature Synthesis)
**Featuretools** applies a set of primitive operations recursively across relational data:

```python
import featuretools as ft

# EntitySet: define tables and relationships
es = ft.EntitySet()
es.add_dataframe(dataframe=customers_df, dataframe_name="customers", index="id")

# Run DFS — generates hundreds of features automatically
feature_matrix, features = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric"],
    agg_primitives=["mean", "std", "count", "max", "min"]
)
```

### Dimensionality Reduction
| Method | Type | Preserves |
|--------|------|-----------|
| PCA | Linear | Maximum variance |
| t-SNE | Non-linear | Local structure (visualization only) |
| UMAP | Non-linear | Both local & global |
| LDA | Supervised linear | Class separability |

---

## 3. Algorithm Selection

AutoML searches over a predefined **search space** of algorithms. Typical search spaces for tabular data:

### Classification Search Space
```
Linear models:    LogisticRegression, SGDClassifier, LinearSVC
Tree-based:       DecisionTree, RandomForest, ExtraTrees, GradientBoosting
Boosting:         XGBoost, LightGBM, CatBoost, AdaBoost
Probabilistic:    GaussianNaive Bayes, BernoulliNB
Neighbours:       KNeighborsClassifier
SVM:              SVC (RBF kernel)
```

### Regression Search Space
```
Linear models:    LinearRegression, Ridge, Lasso, ElasticNet
Tree-based:       DecisionTree, RandomForest, ExtraTrees, GradientBoosting
Boosting:         XGBoost, LightGBM, CatBoost
Neighbours:       KNeighborsRegressor
SVM:              SVR
```

### Algorithm Selection Strategies
1. **Enumerate all** — try every algorithm (expensive but thorough)
2. **Meta-learning warm start** — use historically best algorithms first
3. **Bandit-based** — allocate more budget to promising algorithms
4. **Portfolio initialization** — select diverse algorithms to cover the space

---

## 4. Hyperparameter Optimization (HPO)

See `03_hyperparameter_optimization.md` for full coverage. Key concept here:

### Hyperparameter Types
| Type | Example | Search method |
|------|---------|---------------|
| Continuous | learning rate `[1e-5, 1.0]` | Bayesian, Random |
| Integer | n_estimators `[10, 1000]` | Bayesian (integer), Random |
| Categorical | kernel `{rbf, linear, poly}` | Grid, Random, TPE |
| Conditional | C (only for SVM) | Conditional search space |

### Conditional Search Spaces
AutoML systems model **conditional hyperparameters** — parameters that only exist if a parent parameter has a certain value:

```
Algorithm = RandomForest
    └── n_estimators: [10, 1000]
    └── max_depth: [1, 50] or None
    └── criterion: {gini, entropy}

Algorithm = SVM
    └── kernel: {rbf, linear, poly}
    └── C: [0.001, 1000]             # always active
    └── gamma: [1e-5, 10]            # only active if kernel = rbf or poly
    └── degree: [2, 5]               # only active if kernel = poly
```

---

## 5. Ensembling

AutoML systems often build ensembles from the top-performing models found during search.

### Ensemble Methods

| Method | Description | Strength |
|--------|-------------|----------|
| Voting (hard/soft) | Average predictions | Simple, robust |
| Bagging | Train same model on bootstrap samples | Reduces variance |
| Stacking | Train meta-model on base model predictions | High performance |
| Greedy ensemble selection | Greedily add models that improve ensemble | Auto-sklearn's approach |

### Auto-sklearn Ensemble Construction
Auto-sklearn uses **greedy ensemble selection** (Caruana et al., 2004):

1. Start with the best single model
2. Iteratively add the model that most improves ensemble performance on validation set
3. Allow models to be added multiple times (weighting)
4. Stop when no improvement is possible

This typically yields **2–5% better performance** than the single best model.

---

## 6. Evaluation & Validation

### Cross-Validation Strategies
| Strategy | Description | When to use |
|----------|-------------|-------------|
| K-Fold CV | Split into k folds, rotate | Default choice |
| Stratified K-Fold | Preserves class ratios | Imbalanced classification |
| Time Series Split | Forward-chaining | Time series data |
| Nested CV | Inner CV for HPO, outer for evaluation | Small datasets |
| Holdout | Simple train/val/test split | Large datasets |

### Metrics per Task Type
**Classification**:
- `accuracy` — balanced datasets
- `roc_auc` — probabilistic ranking
- `f1_weighted` — imbalanced datasets
- `log_loss` — probabilistic calibration

**Regression**:
- `r2` — proportion of explained variance
- `rmse` — root mean squared error (penalizes large errors)
- `mae` — mean absolute error (robust to outliers)
- `mape` — percentage error (scale-independent)

---

## Pipeline Component Summary

| Component | Input | Output | AutoML approach |
|-----------|-------|--------|-----------------|
| Preprocessing | Raw data | Clean numerical matrix | Try multiple imputers, encoders, scalers |
| Feature Eng. | Clean matrix | Engineered feature set | Generate + select features |
| Algorithm Selection | Feature set | Fitted model | Search over model zoo |
| HPO | Algorithm + hyperparameters | Optimized model | Bayesian Opt / Hyperband |
| Ensembling | Set of models | Ensemble prediction | Greedy selection / stacking |
