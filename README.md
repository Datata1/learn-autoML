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

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install core dependencies
uv sync

# Install PyCaret (optional, large install)
uv sync --extra pycaret

# Install Auto-sklearn (Linux/macOS only, requires swig)
uv sync --extra autosklearn

# Install visualization extras (Plotly/Kaleido for Optuna plots)
uv sync --extra viz

# Run a script
uv run python 05_implementation_example.py

# Open Jupyter (after uv sync --group dev)
uv run jupyter notebook
```

> **Note**: `auto-sklearn` requires Linux or macOS with `swig` (`sudo apt install swig`). On Windows, use PyCaret or Optuna instead.
