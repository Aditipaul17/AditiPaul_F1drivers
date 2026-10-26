"""Streamlit GUI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import f1_pipeline as P
import f1_plots as G

st.set_page_config(page_title="F1 Driver Performance Predictor", page_icon="🏎️", layout="wide")


@st.cache_data(show_spinner="Scraping Wikipedia & cleaning data...")
def load(refresh=False):
    return P.get_data(refresh)


@st.cache_resource(show_spinner="Training 5 regression models...")
def train(target, _df, n_rows):
    return P.train_models(_df, target)


st.title("🏎️ F1 Driver Performance Prediction")
with st.sidebar:
    st.header("Settings")
    refresh = st.button("🔄 Re-scrape Wikipedia")
    target = st.selectbox("Target to predict", ["Wins", "Podiums", "Poles"])
    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)

if refresh:
    load.clear(); train.clear()
try:
    raw, df = load(refresh)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

tabs = st.tabs(["📋 Data", "📊 EDA", "🎲 Sampling & CLT", "📈 Simple LR", "📉 Multiple LR",
                "🧪 Hypothesis Tests", "🏆 Model Comparison", "🔮 Predict"])

with tabs[0]:
    st.write(f"Raw: {raw.shape} → Clean: {df.shape}")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download clean CSV", df.to_csv(index=False), "f1_clean.csv")
    st.dataframe(df.describe().T)

with tabs[1]:
    c1, c2 = st.columns(2)
    c1.pyplot(G.distributions(df)); c2.pyplot(G.boxplot_era(df))
    c1.pyplot(G.top_nationalities(df))
    cov, pear, spear = P.cov_corr(df)
    c2.pyplot(G.heatmap(pear, "Pearson correlation"))
    c1.pyplot(G.heatmap(spear, "Spearman correlation"))
    with c2:
        st.write("Covariance matrix"); st.dataframe(cov.round(2))

with tabs[2]:
    col = st.selectbox("Variable", ["Wins", "Starts", "Podiums", "Poles"])
    n = st.slider("Sample size", 10, 200, 50)
    s = P.draw_sample(df, n, seed=None or 0)
    st.write(f"Sample mean **{s[col].mean():.2f}** vs population mean **{df[col].mean():.2f}**")
    m, lo, hi = P.confidence_interval(df[col])
    st.write(f"95% CI for population mean: **[{lo:.2f}, {hi:.2f}]** (mean {m:.2f})")
    st.pyplot(G.clt_plot(df[col]))

with tabs[3]:
    st.subheader("Simple Linear Regression (one predictor)")
    xcol = st.selectbox("Predictor (X)", ["Starts", "Poles", "Podiums", "Fastest_Laps", "Career_Years"], key="slr_x")
    ycol = st.selectbox("Response (Y)", ["Wins", "Podiums", "Points"], key="slr_y")
    slr = P.simple_linear_regression(df, xcol, ycol)
    st.write(f"**{ycol} = {slr['slope']:.4f} × {xcol} + {slr['intercept']:.3f}** "
             f"| r = {slr['r_value']:.3f} | slope p-value = {slr['p_value_slope']:.2e}")
    st.dataframe(pd.DataFrame([slr["metrics"]]).round(4))
    c1, c2 = st.columns(2)
    c1.pyplot(G.scatter_regression(df, slr))
    c2.pyplot(G.residual_plots(slr["resid"], slr["pred"], "Residuals (test set)"))
    with st.expander("Full OLS summary"):
        st.code(slr["summary"])

with tabs[4]:
    st.subheader("Multiple Linear Regression (several predictors)")
    all_f = [f for f in P.BASE_FEATURES if f != target]
    chosen = st.multiselect("Predictors", all_f, default=all_f)
    if len(chosen) < 2:
        st.warning("Pick at least 2 predictors for multiple regression.")
    else:
        mlr = P.multiple_linear_regression(df, target, chosen)
        m = st.columns(4)
        m[0].metric("R²", f"{mlr['r2']:.3f}"); m[1].metric("Adjusted R²", f"{mlr['adj_r2']:.3f}")
        m[2].metric("F-statistic", f"{mlr['f_stat']:.1f}"); m[3].metric("F p-value", f"{mlr['f_pvalue']:.2e}")
        st.write("**Coefficients**"); st.dataframe(mlr["coefs"].round(4))
        st.write("**Test-set metrics**"); st.dataframe(pd.DataFrame([mlr["metrics"]]).round(4))
        st.pyplot(G.residual_vs_fitted(mlr["resid"], mlr["fitted"]))
        with st.expander("Full OLS summary"):
            st.code(mlr["summary"])

with tabs[5]:
    c = st.columns(3)
    mu_s = c[0].number_input("Z-test: H0 mean Starts =", value=50)
    mu_w = c[1].number_input("t-test: H0 mean Wins =", value=2)
    n_s = c[2].number_input("Z-test sample size", 30, 300, 100)
    tests = P.run_statistical_tests(df, alpha, mu_s, mu_w, int(n_s))
    st.dataframe(tests, use_container_width=True)

with tabs[6]:
    res = train(target, df, len(df))
    st.success(f"Best model for **{target}**: {res['best_name']}")
    st.dataframe(res["results"].round(4), use_container_width=True)
    st.pyplot(G.model_comparison(res["results"]))
    pick = st.selectbox("Inspect model", list(res["models"]), index=list(res["models"]).index(res["best_name"]))
    c1, c2 = st.columns(2)
    c1.pyplot(G.actual_vs_pred(res["y_test"], res["preds"][pick], pick))
    c2.pyplot(G.feature_importance(res["models"][pick], res["features"]))

with tabs[7]:
    res = train(target, df, len(df))
    st.write(f"Enter a driver's career profile to predict **{target}**")
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(res["features"]):
        vals[f] = cols[i % 3].number_input(f, min_value=0.0, value=float(round(df[f].median(), 0)), step=1.0)
    X = pd.DataFrame([vals])
    if st.button("Predict", type="primary"):
        st.metric(f"Best model ({res['best_name']})", f"{max(0, res['best_model'].predict(X)[0]):.1f} {target}")
        st.dataframe(pd.DataFrame({m: [max(0, mod.predict(X)[0])] for m, mod in res["models"].items()},
                                  index=["Prediction"]).round(2))
