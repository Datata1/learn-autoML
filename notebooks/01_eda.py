import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris, load_wine

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({"figure.dpi": 120})
    return load_iris, load_wine, mo, np, pd, plt, sns


@app.cell
def _(mo):
    return mo.md(r"""
    # AutoML — Exploratory Data Analysis

    Interactive exploration of the two benchmark datasets used in this project.
    Switch between datasets using the dropdown below.
    """)


@app.cell
def _(mo):
    selector = mo.ui.dropdown(
        options={
            "Iris  (150 × 4 features, 3 classes)": "iris",
            "Wine  (178 × 13 features, 3 classes)": "wine",
        },
        value="iris",
        label="Dataset",
    )
    return (selector,)


# ── Load data (reactive to selector) ─────────────────────────────────────────
@app.cell
def _(load_iris, load_wine, pd, selector):
    _raw = load_iris() if selector.value == "iris" else load_wine()

    feature_names = list(_raw.feature_names)
    target_names = list(_raw.target_names)

    df = pd.DataFrame(_raw.data, columns=feature_names)
    df["target"] = _raw.target
    df["class"] = [target_names[i] for i in _raw.target]
    return df, feature_names, target_names


# ── Summary stats ─────────────────────────────────────────────────────────────
@app.cell
def _(df, feature_names, mo, target_names):
    _counts = df["class"].value_counts().to_dict()
    _counts_str = "  |  ".join(f"**{k}**: {v}" for k, v in _counts.items())
    return mo.md(f"""
    ## Dataset Summary

    | Metric | Value |
    |---|---|
    | Samples | {len(df)} |
    | Features | {len(feature_names)} |
    | Classes | {len(target_names)} ({", ".join(target_names)}) |
    | Missing values | {df.isnull().sum().sum()} |
    | Class distribution | {_counts_str} |
    """)


# ── Class distribution + describe table ──────────────────────────────────────
@app.cell
def _(df, plt, target_names):
    _colors = ["#4c72b0", "#dd8452", "#55a868"]
    _counts = df["class"].value_counts()

    fig_overview, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Bar chart
    bars = axes[0].bar(
        _counts.index, _counts.values,
        color=_colors[: len(target_names)], edgecolor="white", linewidth=1.5,
    )
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Samples")
    for bar, v in zip(bars, _counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, v + 0.4,
            str(v), ha="center", va="bottom", fontsize=10,
        )

    # Describe table
    _desc = (
        df.drop(["target", "class"], axis=1)
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(2)
    )
    _col_labels = [c.split(" (")[0][:12] for c in _desc.columns]
    t = axes[1].table(
        cellText=_desc.values,
        rowLabels=_desc.index,
        colLabels=_col_labels,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    axes[1].axis("off")
    axes[1].set_title("Feature Statistics", fontweight="bold")

    plt.tight_layout()
    return (fig_overview,)


# ── Violin plots per feature and class ───────────────────────────────────────
@app.cell
def _(df, mo):
    return mo.md("## Feature Distributions per Class")


@app.cell
def _(df, plt, target_names):
    _features = [c for c in df.columns if c not in ("target", "class")]
    _n = len(_features)
    _ncols = min(4, _n)
    _nrows = (_n + _ncols - 1) // _ncols

    fig_violin, axes = plt.subplots(
        _nrows, _ncols, figsize=(_ncols * 3.2, _nrows * 3.5), squeeze=False
    )
    _flat = axes.flatten()

    _colors = ["#4c72b0", "#dd8452", "#55a868"]

    for i, feat in enumerate(_features):
        ax = _flat[i]
        data_by_class = [df[df["class"] == cls][feat].values for cls in target_names]
        parts = ax.violinplot(
            data_by_class,
            positions=range(len(target_names)),
            showmedians=True,
            showextrema=True,
        )
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(_colors[j % len(_colors)])
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=15, ha="right", fontsize=8)
        ax.set_title(feat.split(" (")[0], fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)

    for i in range(_n, len(_flat)):
        _flat[i].set_visible(False)

    fig_violin.suptitle("Feature Distributions per Class (Violin Plots)", fontweight="bold")
    plt.tight_layout()
    return (fig_violin,)


# ── Correlation heatmap ───────────────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Feature Correlation Matrix")


@app.cell
def _(df, plt, sns):
    _num = df.drop(["target", "class"], axis=1)
    _corr = _num.corr()
    _n = len(_corr)

    fig_corr, ax = plt.subplots(figsize=(max(6, _n * 0.85), max(5, _n * 0.75)))
    sns.heatmap(
        _corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": max(6, 10 - _n // 3)},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    return (fig_corr,)


# ── Pairplot ──────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Pairplot (first 5 features)")


@app.cell
def _(df, sns):
    _features = [c for c in df.columns if c not in ("target", "class")]
    _plot_cols = _features[:5] + ["class"]

    _g = sns.pairplot(
        df[_plot_cols],
        hue="class",
        diag_kind="kde",
        plot_kws={"alpha": 0.55, "s": 25},
        diag_kws={"linewidth": 1.5},
        palette="Set2",
    )
    _g.fig.suptitle("Pairplot — Class Separability", y=1.02, fontweight="bold")
    return (_g.fig,)


# ── Box plots (outlier focus) ─────────────────────────────────────────────────
@app.cell
def _(mo):
    return mo.md("## Outlier Overview (Box Plots)")


@app.cell
def _(df, plt, sns):
    _num = df.drop(["target", "class"], axis=1)
    _features = list(_num.columns)
    _ncols = min(4, len(_features))
    _nrows = (len(_features) + _ncols - 1) // _ncols

    fig_box, axes = plt.subplots(
        _nrows, _ncols, figsize=(_ncols * 3, _nrows * 3), squeeze=False
    )
    _flat = axes.flatten()

    for i, feat in enumerate(_features):
        _data = [
            df[df["class"] == cls][feat].values for cls in df["class"].unique()
        ]
        _flat[i].boxplot(_data, labels=df["class"].unique(), patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
        _flat[i].set_title(feat.split(" (")[0], fontsize=9)
        _flat[i].tick_params(axis="x", rotation=15, labelsize=8)

    for i in range(len(_features), len(_flat)):
        _flat[i].set_visible(False)

    fig_box.suptitle("Box Plots per Class (outliers shown as circles)", fontweight="bold")
    plt.tight_layout()
    return (fig_box,)
