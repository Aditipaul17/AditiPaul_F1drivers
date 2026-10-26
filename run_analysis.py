"""Run the full pipeline from the command line:  python run_analysis.py [--refresh]"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import pandas as pd

import f1_pipeline as P
import f1_plots as G

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 30)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")


def main(refresh=False):
    raw, df = P.get_data(refresh)
    print(f"Raw shape {raw.shape} -> clean shape {df.shape}")
    print(df.head(), "\n"); print(df.describe().T, "\n")
    print("Missing values:\n", df.isna().sum()[lambda s: s > 0], "\n")

    # EDA
    save(G.distributions(df), "01_distributions")
    save(G.boxplot_era(df), "02_win_rate_by_era")
    save(G.top_nationalities(df), "03_nationalities")
    cov, pear, spear = P.cov_corr(df)
    print("Covariance:\n", cov.round(2), "\n\nPearson:\n", pear.round(2), "\n")
    save(G.heatmap(pear, "Pearson"), "04_pearson_heatmap")
    save(G.heatmap(spear, "Spearman"), "05_spearman_heatmap")

    # Sampling / CLT / CI
    s = P.draw_sample(df, 50)
    print(f"Sample(50) mean Wins={s['Wins'].mean():.2f} vs population {df['Wins'].mean():.2f}")
    save(G.clt_plot(df["Wins"]), "06_clt")
    m, lo, hi = P.confidence_interval(df["Wins"])
    print(f"95% CI for mean Wins: {m:.2f} [{lo:.2f}, {hi:.2f}]\n")

    # Simple linear regression (1 predictor)
    slr = P.simple_linear_regression(df)
    print("=== SIMPLE LINEAR REGRESSION (Starts -> Wins) ===")
    print(f"Wins = {slr['slope']:.4f} * Starts + {slr['intercept']:.3f}  (p={slr['p_value_slope']:.2e})")
    print({k: round(v, 4) for k, v in slr["metrics"].items()}, "\n")
    save(G.scatter_regression(df, slr), "07_simple_lr")
    save(G.residual_plots(slr["resid"], slr["pred"], "Simple LR residuals"), "07b_simple_lr_residuals")

    # Multiple linear regression (several predictors)
    mlr = P.multiple_linear_regression(df, "Wins")
    print("=== MULTIPLE LINEAR REGRESSION ===")
    print(mlr["summary"])
    print("\nTest metrics:", {k: round(v, 4) for k, v in mlr["metrics"].items()},
          f"\nR2={mlr['r2']:.3f}  Adj R2={mlr['adj_r2']:.3f}  F={mlr['f_stat']:.1f} (p={mlr['f_pvalue']:.2e})\n")
    save(G.residual_vs_fitted(mlr["resid"], mlr["fitted"]), "07c_mlr_residuals")

    # Hypothesis tests
    tests = P.run_statistical_tests(df)
    print(tests.to_string(index=False), "\n")
    tests.to_csv(OUT / "hypothesis_tests.csv", index=False)

    # Model comparison
    res = P.train_models(df, "Wins")
    print(res["results"].drop(columns="Best_Params").round(4).to_string(index=False))
    print(f"\n*** BEST MODEL: {res['best_name']} ***")
    res["results"].to_csv(OUT / "model_comparison.csv", index=False)
    save(G.model_comparison(res["results"]), "08_model_comparison")
    save(G.actual_vs_pred(res["y_test"], res["preds"][res["best_name"]], res["best_name"]), "09_best_actual_vs_pred")
    save(G.feature_importance(res["best_model"], res["features"]), "10_feature_importance")
    joblib.dump({"model": res["best_model"], "features": res["features"], "name": res["best_name"]},
                OUT / "best_model.joblib")
    print(f"\nSaved plots/tables/model in {OUT}")


if __name__ == "__main__":
    main("--refresh" in sys.argv)
