"""Plot helpers shared by run_analysis.py and the Streamlit app."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")


def distributions(df, cols=("Starts", "Wins", "Podiums", "Poles")):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], kde=True, ax=ax, color="#e10600")
        ax.set_title(f"Distribution of {c}")
    fig.tight_layout()
    return fig


def boxplot_era(df, y="Win_Rate"):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["1950-69", "1970-89", "1990-2009", "2010+"]
    sns.boxplot(data=df[df["Starts"] >= 20], x="Era", y=y, order=order, ax=ax, palette="Reds")
    ax.set_title(f"{y} by debut era (≥20 starts)")
    return fig


def top_nationalities(df, n=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = df["Nationality"].value_counts().head(n)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="Reds_r")
    ax.set_title(f"Top {n} nationalities by number of drivers")
    ax.set_xlabel("Drivers")
    return fig


def heatmap(corr, title="Correlation"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    return fig


def scatter_regression(df, slr):
    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = slr["x"], slr["y"]
    ax.scatter(df[x], df[y], alpha=.5, color="#555")
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xs, slr["model"].predict(xs.reshape(-1, 1)), color="#e10600", lw=2,
            label=f"y = {slr['slope']:.3f}x + {slr['intercept']:.2f}")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(); ax.set_title("Simple Linear Regression")
    return fig


def clt_plot(series, sizes=(2, 10, 30, 100)):
    from f1_pipeline import sampling_distribution
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(16, 3.6))
    sns.histplot(series, ax=axes[0], color="grey", bins=30)
    axes[0].set_title(f"Population ({series.name})")
    for ax, n in zip(axes[1:], sizes):
        means = sampling_distribution(series, n=n)
        sns.histplot(means, kde=True, ax=ax, color="#e10600", bins=30)
        ax.set_title(f"Sample means n={n}")
    fig.tight_layout()
    return fig


def qq_plot(series):
    fig, ax = plt.subplots(figsize=(5, 4))
    stats.probplot(series, dist="norm", plot=ax)
    return fig


def model_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, c in zip(axes, ["RMSE", "MAE", "R2"], ["#e10600", "#f5a623", "#1f77b4"]):
        sns.barplot(data=results, x="Model", y=m, ax=ax, color=c)
        ax.set_title(m); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def actual_vs_pred(y_test, pred, name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, pred, alpha=.6, color="#e10600")
    lim = [0, max(max(y_test), max(pred))]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"{name}: actual vs predicted")
    return fig


def feature_importance(model, features):
    m = model.named_steps["m"]
    vals = getattr(m, "feature_importances_", None)
    if vals is None:
        vals = np.abs(m.coef_)
    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(vals)
    ax.barh(np.array(features)[order], np.array(vals)[order], color="#e10600")
    ax.set_title("Feature importance / |coefficient|")
    return fig


def residual_plots(resid, fitted, title="Residual diagnostics"):
    from scipy import stats as st
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(fitted, resid, alpha=.5, color="#e10600"); axes[0].axhline(0, color="k", ls="--")
    axes[0].set_xlabel("Fitted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs fitted")
    sns.histplot(resid, kde=True, ax=axes[1], color="#e10600"); axes[1].set_title("Residual distribution")
    st.probplot(resid, dist="norm", plot=axes[2]); axes[2].set_title("Q-Q plot")
    fig.suptitle(title); fig.tight_layout()
    return fig


def residual_vs_fitted(resid, fitted, title="MLR: residuals vs fitted"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(fitted, resid, alpha=.5, color="#e10600")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals"); ax.set_title(title)
    return fig
