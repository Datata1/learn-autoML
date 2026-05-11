# AutoML Foundations

## 1. What is AutoML?

**Automated Machine Learning (AutoML)** is the process of automating the end-to-end application of machine learning to real-world problems. It aims to make ML accessible to non-experts while accelerating the work of experts.

A classical ML workflow requires a human to:
1. Understand and preprocess the data
2. Select a suitable algorithm
3. Tune hyperparameters
4. Evaluate and iterate

AutoML systems automate all or parts of this process.

### Formal Definition

Given a dataset `D = (X_train, y_train, X_test)` and a performance metric `f`, AutoML finds a pipeline `P*` such that:

```
P* = argmax_{P ∈ P_space} f(P, D_train, D_val)
```

where `P_space` is the space of all valid ML pipelines (preprocessors + algorithms + hyperparameters).

---

## 2. History & Milestones

| Year | Milestone |
|------|-----------|
| 2004 | Early algorithm selection research (Rice's framework) |
| 2012 | **AutoWEKA** — first full AutoML system (Bayesian Opt over WEKA) |
| 2015 | **Auto-sklearn** — sklearn-based, won AutoML challenge |
| 2016 | **TPOT** — genetic programming for pipeline optimization |
| 2017 | **Google AutoML** (cloud product), **Neural Architecture Search** gains traction |
| 2019 | **AutoKeras**, **H2O AutoML** mature; NAS papers like DARTS |
| 2020 | **PyCaret** 2.0 — democratization push; **TabNet** (attention for tabular) |
| 2022–25 | LLMs start assisting AutoML (meta-learning from descriptions); **Auto-sklearn 2.0** |

---

## 3. The No-Free-Lunch Theorem (Motivation)

> "Any two optimization algorithms are equivalent when their performance is averaged across all possible problems." — Wolpert & Macready, 1997

**Implication for ML**: No single algorithm works best on all datasets. You must select and configure models per dataset. AutoML automates this selection process systematically rather than by intuition.

---

## 4. AutoML vs. Classical ML

| Aspect | Classical ML | AutoML |
|--------|-------------|--------|
| Algorithm selection | Human-driven | Automated search |
| Hyperparameter tuning | Manual / grid search | Bayesian Opt, Hyperband, etc. |
| Feature engineering | Domain expert | Automated (partial) |
| Time investment | Days/weeks | Hours |
| Interpretability | High (you built it) | Lower (black-box search) |
| Reproducibility | Manual tracking | Built-in (most frameworks) |
| Required expertise | High | Low to medium |
| Flexibility | Full | Constrained by search space |

---

## 5. The CASH Problem

**Combined Algorithm Selection and Hyperparameter Optimization (CASH)** is the core problem AutoML solves.

Formally, given:
- A set of algorithms `A = {A1, A2, ..., An}`
- Each algorithm `Ai` has a hyperparameter space `Λi`
- A dataset split into train/validation

Find:

```
(A*, λ*) = argmax_{Ai ∈ A, λ ∈ Λi} Validation_Score(Ai(λ), D_train, D_val)
```

This is hard because:
- The search space is **combinatorial** (discrete algorithm choices)
- Each evaluation is **expensive** (training a model)
- The space is **hierarchical** (different algorithms have different hyperparameters)

---

## 6. What AutoML Can and Cannot Do

### AutoML excels at:
- Tabular data (structured, numerical, categorical)
- Standard classification and regression tasks
- Reducing time from data to first good model
- Baseline model generation
- Hyperparameter tuning at scale

### AutoML struggles with:
- Highly domain-specific feature engineering (medical imaging, NLP quirks)
- Very small datasets (< 100 samples) — overfitting to search
- Concept drift and non-stationary data
- Explaining *why* a model works
- Novel architectures / research frontiers
- Data collection and labeling

---

## 7. Meta-Learning (Learning to Learn)

Meta-learning is the technique of using past experience across datasets to warm-start the AutoML search.

**How it works**:
1. Collect meta-features from `D_new` (number of samples, number of features, class imbalance, skewness, ...)
2. Find historically similar datasets in a meta-database
3. Use the best configurations from those datasets as starting points

**Effect**: The search converges faster because it doesn't start from scratch.

Auto-sklearn uses meta-learning to initialize its Bayesian Optimization with a set of promising configurations.

---

## 8. Types of AutoML Systems

### Full-Pipeline AutoML
Automates preprocessing + algorithm selection + HPO + ensembling.
- Examples: Auto-sklearn, TPOT, H2O AutoML, PyCaret

### HPO-only Libraries
Focus only on hyperparameter optimization; you choose the algorithm.
- Examples: Optuna, Hyperopt, Ray Tune, Ax

### Neural Architecture Search (NAS)
Automates the design of neural network architectures.
- Examples: AutoKeras, Neural Network Intelligence (NNI), DARTS

### Cloud AutoML
Fully managed, no code required.
- Examples: Google AutoML, AWS SageMaker Autopilot, Azure AutoML

---

## Key Takeaways

- AutoML automates the **CASH problem**: algorithm selection + hyperparameter optimization
- The **No-Free-Lunch theorem** makes this necessary — no single algorithm wins everywhere
- **Meta-learning** makes AutoML faster by leveraging historical knowledge
- AutoML is not a replacement for domain expertise, but a powerful accelerator
