import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import optuna
    import pandas as pd

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    plt.rcParams.update({"figure.dpi": 120})

    RESULTS_DIR = Path(__file__).parent.parent / "results"
    return RESULTS_DIR, json, mo, mpatches, np, optuna, pd, plt


# ── Guard: results must exist ─────────────────────────────────────────────────
@app.cell
def _(RESULTS_DIR, mo):
    _results_file = RESULTS_DIR / "experiments.json"
    mo.stop(
        not _results_file.exists(),
        mo.callout(
            mo.md(
                "**No results found.**\n\n"
                "Run the experiment script first:\n"
                "```\nuv run python run_experiments.py\n```"
            ),
            kind="warn",
        ),
    )
    return


@app.cell
def _(mo):
    return mo.md(r"""
    # AutoML — Experiment Results

    Comparison of manual baselines vs. Optuna HPO vs. TPOT across Iris and Wine.
    """)


# ── Load results ──────────────────────────────────────────────────────────────
@app.cell
def _(RESULTS_DIR, json):
    with open(RESULTS_DIR / "experiments.json") as _f:
        results = json.load(_f)
    return (results,)


# ── Dataset selector ─────────────────────────────────────────────────────────
@app.cell
def _(mo):
    dataset_sel = mo.ui.dropdown(
        options={"Iris": "iris", "Wine": "wine"},
        value="iris",
        label="Dataset",
    )
    return (dataset_sel,)


# ── Derive per-dataset data ───────────────────────────────────────────────────
@app.cell
def _(dataset_sel, pd, results):
    _ds = dataset_sel.value
    _r = results[_ds]

    # Build flat comparison table
    rows = []
    for model_name, metrics in _r["baselines"].items():
        rows.append({
            "Method": model_name,
            "Type": "Baseline",
            "CV Accuracy": metrics["cv_mean"],
            "CV ± std": metrics["cv_std"],
            "Test Accuracy": metrics["test"],
            "Time (s)": metrics["time_s"],
        })

    _opt = _r["optuna"]
    rows.append({
        "Method": f"Optuna ({_opt['best_params'].get('algo', '?').upper()})",
        "Type": "AutoML",
        "CV Accuracy": _opt["cv_score"],
        "CV ± std": None,
        "Test Accuracy": _opt["test"],
        "Time (s)": _opt["time_s"],
    })

    if _r.get("tpot"):
        rows.append({
            "Method": "TPOT",
            "Type": "AutoML",
            "CV Accuracy": None,
            "CV ± std": None,
            "Test Accuracy": _r["tpot"]["test"],
            "Time (s)": _r["tpot"]["time_s"],
        })

    comparison_df = pd.DataFrame(rows)
    best_baseline = comparison_df[comparison_df["Type"] == "Baseline"]["Test Accuracy"].max()
    return best_baseline, comparison_df


# ── Leaderboard table ─────────────────────────────────────────────────────────
@app.cell
def _(comparison_df, mo):
    return mo.md("## Leaderboard"), mo.table(
        comparison_df.sort_values("Test Accuracy", ascending=False)
        .reset_index(drop=True)
        .round(4)
    )


# ── Accuracy comparison bar chart ────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Accuracy: Baseline vs. AutoML")


@app.cell
def _(best_baseline, comparison_df, mpatches, plt):
    _COLORS = {"Baseline": "#4c72b0", "AutoML": "#dd8452"}

    _df = comparison_df.sort_values("Test Accuracy", ascending=True)
    _colors = [_COLORS[t] for t in _df["Type"]]

    fig_bar, ax = plt.subplots(figsize=(10, max(4, len(_df) * 0.55)))
    bars = ax.barh(_df["Method"], _df["Test Accuracy"], color=_colors, edgecolor="white")
    ax.axvline(best_baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        best_baseline + 0.001, -0.6,
        f"best baseline\n{best_baseline:.4f}",
        color="gray", fontsize=8, va="top",
    )

    _xmin = max(0, _df["Test Accuracy"].min() - 0.04)
    ax.set_xlim(_xmin, min(1.03, _df["Test Accuracy"].max() + 0.04))

    for bar, v in zip(bars, _df["Test Accuracy"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Test Accuracy")
    ax.set_title("Test Accuracy Comparison", fontweight="bold")
    legend_handles = [
        mpatches.Patch(color=c, label=l) for l, c in _COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    return (fig_bar,)


# ── Optuna optimization history ───────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Optuna: Optimization History")


@app.cell
def _(RESULTS_DIR, dataset_sel, mo, optuna, plt):
    _study_name = f"{dataset_sel.value}_automl"
    _storage = f"sqlite:///{RESULTS_DIR}/optuna.db"

    _db_path = RESULTS_DIR / "optuna.db"
    mo.stop(
        not _db_path.exists(),
        mo.callout(mo.md("Optuna database not found — run experiments first."), kind="warn"),
    )

    _study = optuna.load_study(study_name=_study_name, storage=_storage)
    _trials_df = _study.trials_dataframe()

    fig_history, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Objective over trials
    axes[0].scatter(
        _trials_df["number"],
        _trials_df["value"],
        c=_trials_df["value"],
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    _best_so_far = _trials_df["value"].cummax()
    axes[0].plot(_trials_df["number"], _best_so_far, color="#dd8452", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("CV Accuracy")
    axes[0].set_title("Optimization History", fontweight="bold")
    axes[0].legend()

    # Algorithm distribution
    _algo_col = "params_algo"
    if _algo_col in _trials_df.columns:
        _algo_counts = _trials_df[_algo_col].value_counts()
        _best_per_algo = (
            _trials_df.groupby(_algo_col)["value"].max().reindex(_algo_counts.index)
        )
        _x = range(len(_algo_counts))
        axes[1].bar(_x, _algo_counts.values, color="#4c72b0", alpha=0.7, label="# trials")
        ax2 = axes[1].twinx()
        ax2.plot(_x, _best_per_algo.values, "o-", color="#dd8452", linewidth=2, label="Best CV")
        axes[1].set_xticks(list(_x))
        axes[1].set_xticklabels(_algo_counts.index)
        axes[1].set_ylabel("Number of trials")
        ax2.set_ylabel("Best CV Accuracy", color="#dd8452")
        axes[1].set_title("Trials & Best Score per Algorithm", fontweight="bold")

    plt.tight_layout()
    return fig_history, _study


# ── Optuna parameter importances ──────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Optuna: Hyperparameter Importance")


@app.cell
def _(_study, mo, plt):
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        fig_imp = plot_param_importances(_study)
        plt.tight_layout()
    except Exception as e:
        mo.stop(True, mo.callout(mo.md(f"Could not compute importances: {e}"), kind="warn"))
        fig_imp = None
    return (fig_imp,)


# ── Best configuration detail ─────────────────────────────────────────────────
@app.cell
def _(dataset_sel, mo, results):
    _opt = results[dataset_sel.value]["optuna"]
    _rows = [{"Parameter": k, "Value": v} for k, v in _opt["best_params"].items()]

    return mo.vstack([
        mo.md("## Best Configuration (Optuna)"),
        mo.hstack([
            mo.stat(
                value=f"{_opt['cv_score']:.4f}",
                label="Best CV accuracy",
                caption="5-fold cross-validation",
            ),
            mo.stat(
                value=f"{_opt['test']:.4f}",
                label="Test accuracy",
                caption="Held-out 20% split",
            ),
            mo.stat(
                value=str(_opt["n_trials"]),
                label="Trials",
                caption="",
            ),
            mo.stat(
                value=f"{_opt['time_s']:.0f}s",
                label="Total search time",
                caption="",
            ),
        ]),
        mo.table(_rows),
    ])


# ── Cross-dataset comparison ──────────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Cross-Dataset Summary")


@app.cell
def _(mo, pd, results):
    _rows = []
    for _ds, _r in results.items():
        _best_bl = max(v["test"] for v in _r["baselines"].values())
        _opt_test = _r["optuna"]["test"]
        _tpot_test = (_r["tpot"] or {}).get("test")
        _rows.append({
            "Dataset": _ds.capitalize(),
            "Best Baseline": _best_bl,
            "Optuna": _opt_test,
            "TPOT": _tpot_test,
            "Improvement (Optuna vs Best BL)": round(_opt_test - _best_bl, 4),
        })

    return mo.table(pd.DataFrame(_rows).round(4))
