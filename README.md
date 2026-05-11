# AutoML — University Project Knowledge Base

> **Purpose**: Deep-dive reference for learning AutoML and a structured skill set for Claude Code agents.
> **Language**: English
> **Datasets**: Iris & Wine (sklearn built-ins)

---

## Learning Path

Read the files in this order for a structured understanding:

| Step | File | Topic | Time |
|------|------|--------|------|
| 1 | [01_foundations.md](01_foundations.md) | What is AutoML, history, motivation | ~20 min |
| 2 | [02_pipeline_components.md](02_pipeline_components.md) | Every stage of an AutoML pipeline | ~30 min |
| 3 | [03_hyperparameter_optimization.md](03_hyperparameter_optimization.md) | HPO algorithms in depth | ~30 min |
| 4 | [04_frameworks_comparison.md](04_frameworks_comparison.md) | Framework landscape & how to choose | ~20 min |
| 5 | [05_implementation_example.md](05_implementation_example.md) | Hands-on code with Iris & Wine | ~45 min |

---

## Quick-Reference Concept Map

```
AutoML
│
├── Problem Formulation
│   └── CASH Problem (Combined Algorithm Selection & Hyperparameter Optimization)
│
├── Pipeline Components
│   ├── Data Preprocessing       → imputation, encoding, scaling
│   ├── Feature Engineering      → selection, generation, transformation
│   ├── Algorithm Selection      → which ML model family to use
│   ├── Hyperparameter Opt.      → tuning the chosen model
│   └── Ensembling               → stacking multiple models
│
├── HPO Methods
│   ├── Grid Search              → exhaustive, expensive
│   ├── Random Search            → faster, surprisingly effective
│   ├── Bayesian Optimization    → probabilistic model of objective
│   ├── Hyperband / ASHA         → early stopping + random search
│   └── Population-based         → evolutionary strategies
│
├── Neural AutoML
│   └── NAS (Neural Architecture Search)
│       ├── Evolutionary / RL-based
│       ├── Differentiable (DARTS)
│       └── One-shot / Weight Sharing
│
└── Frameworks
    ├── Auto-sklearn   → Bayesian Opt + meta-learning, sklearn API
    ├── TPOT           → genetic programming, pipeline evolution
    ├── PyCaret        → low-code, broad coverage
    ├── H2O AutoML     → scalable, production-ready
    ├── AutoKeras      → NAS for Keras/TF models
    └── Optuna         → flexible HPO library (not full AutoML)
```

---

## Agent Skills

For Claude Code agents, the dedicated skill file is at:

```
.claude/skills/automl_agent.md
```

This file contains structured instructions for implementing AutoML tasks programmatically.

---

## Workflow

```bash
# 1 — install dependencies
uv sync

# 2 — run all experiments (saves to results/)
uv run python run_experiments.py
uv run python run_experiments.py --no-tpot   # faster, skips TPOT

# 3 — open notebooks
uv run marimo edit notebooks/01_eda.py       # EDA
uv run marimo edit notebooks/02_results.py  # Results & HPO visualizations
```

Optional extras:
```bash
uv sync --extra tpot        # TPOT genetic pipeline search
uv sync --extra pycaret     # PyCaret low-code AutoML
uv sync --extra viz         # Plotly + Kaleido for extra Optuna plots
```

> **Note on auto-sklearn**: requires Linux/macOS + `swig` (`sudo apt install swig`). Not supported on Python 3.13 yet — use PyCaret or Optuna instead.
