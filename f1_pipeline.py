"""
F1 Driver Performance Prediction - data, statistics and modelling pipeline.
Covers: scraping, cleaning, sampling/CLT, regression, hypothesis tests, model comparison.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_drivers"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational F1 regression project)"}
CURRENT_YEAR = 2026
DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "f1_drivers_raw.csv"
CLEAN_CSV = DATA_DIR / "f1_drivers_clean.csv"
SEED = 42

# keyword in scraped header -> clean column name (order matters)
COLUMN_MAP = [
    ("championship", "Championships"),
    ("entries", "Entries"),
    ("starts", "Starts"),
    ("pole", "Poles"),
    ("wins", "Wins"),
    ("podium", "Podiums"),
    ("fastest", "Fastest_Laps"),
    ("points", "Points"),
    ("seasons", "Seasons"),
    ("nationality", "Nationality"),
    ("driver", "Driver"),
]
NUMERIC_COLS = ["Championships", "Entries", "Starts", "Poles", "Wins",
                "Podiums", "Fastest_Laps", "Points"]


# ---------------------------------------------------------------- 1-3 SCRAPING
def _map_header(h: str):
    h = h.lower()
    for key, name in COLUMN_MAP:
        if key in h:
            return name
    return h


def scrape_f1_drivers(url: str = URL) -> pd.DataFrame:
    """Scrape the main drivers table with requests + BeautifulSoup -> raw DataFrame."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for sup in soup.find_all("sup"):          # remove [1], [a] footnote markers
        sup.decompose()

    best = None
    for table in soup.find_all("table", class_="wikitable"):
        first_row = table.find("tr")
        if first_row is None:
            continue
        heads = [th.get_text(" ", strip=True) for th in first_row.find_all(["th", "td"])]
        low = " ".join(heads).lower()
        if "wins" in low and "starts" in low:
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if len(cells) == len(heads):
                    rows.append(cells)
            df = pd.DataFrame(rows, columns=[_map_header(h) for h in heads])
            if best is None or len(df) > len(best):
                best = df
    if best is None:
        raise RuntimeError("Could not find the drivers table on the page.")
    return best


# --------------------------------------------------------------- 4 CLEANING
def _first_number(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"\d[\d,]*\.?\d*|\.\d+", str(x))
    return float(m.group().replace(",", "")) if m else np.nan


def _parse_seasons(text):
    """'1950-1955, 1958' -> (n_unique_seasons, first, last)."""
    t = str(text).lower().replace("present", str(CURRENT_YEAR))
    years = set()
    for a, b in re.findall(r"(\d{4})\s*[–—-]\s*(\d{4})", t):
        years.update(range(int(a), int(b) + 1))
    t_no_range = re.sub(r"\d{4}\s*[–—-]\s*\d{4}", " ", t)
    years.update(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", t_no_range))
    if not years:
        return pd.Series([np.nan, np.nan, np.nan])
    return pd.Series([len(years), min(years), max(years)])


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Driver"] = (df["Driver"].astype(str)
                    .str.replace(r"[*~^†‡]+", "", regex=True)
                    .str.replace(r"\[.*?\]", "", regex=True).str.strip())
    for c in NUMERIC_COLS:
        if c in df:
            df[c] = df[c].apply(_first_number)
    df[["Season_Count", "First_Season", "Last_Season"]] = df["Seasons"].apply(_parse_seasons)
    df["Career_Years"] = df["Last_Season"] - df["First_Season"] + 1
    df["Nationality"] = df["Nationality"].astype(str).str.strip()

    df = df.drop_duplicates(subset="Driver")
    df = df.dropna(subset=["Starts", "Wins", "Poles", "Podiums", "First_Season"])
    df = df[df["Starts"] > 0].copy()
    for c in ["Championships", "Fastest_Laps", "Points"]:   # missing means zero
        df[c] = df[c].fillna(0)

    # feature engineering
    df["Win_Rate"] = df["Wins"] / df["Starts"]
    df["Podium_Rate"] = df["Podiums"] / df["Starts"]
    df["Pole_Rate"] = df["Poles"] / df["Starts"]
    df["Has_Won"] = np.where(df["Wins"] > 0, "Winner", "Non-winner")
    df["Is_Champion"] = np.where(df["Championships"] > 0, "Champion", "Non-champion")
    df["Era"] = pd.cut(df["First_Season"], [1949, 1969, 1989, 2009, 2100],
                       labels=["1950-69", "1970-89", "1990-2009", "2010+"]).astype(str)
    return df.reset_index(drop=True)


def get_data(refresh: bool = False, url: str = URL):
    """Return (raw, clean). Scrapes the web; falls back to the cached CSV if offline."""
    if refresh or not RAW_CSV.exists():
        try:
            raw = scrape_f1_drivers(url)
            raw.to_csv(RAW_CSV, index=False)
        except Exception as e:
            if RAW_CSV.exists():
                print(f"Scrape failed ({e}); using cached CSV.")
                raw = pd.read_csv(RAW_CSV)
            else:
                raise
    else:
        raw = pd.read_csv(RAW_CSV)
    clean = clean_data(raw)
    clean.to_csv(CLEAN_CSV, index=False)
    return raw, clean


# -------------------------------------------------- 7-9 SAMPLING / CLT
def draw_sample(df, n=50, seed=SEED):
    return df.sample(n=min(n, len(df)), random_state=seed)


def sampling_distribution(series: pd.Series, n=30, reps=1000, seed=SEED):
    """Means of `reps` random samples of size n (for the CLT demo)."""
    rng = np.random.default_rng(seed)
    vals = series.dropna().to_numpy()
    return np.array([rng.choice(vals, n, replace=True).mean() for _ in range(reps)])


# ------------------------------------------------- 10-11 SIMPLE LINEAR REGRESSION
def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def simple_linear_regression(df, x="Starts", y="Wins"):
    """SIMPLE linear regression: ONE predictor.  y = b0 + b1*x"""
    X_tr, X_te, y_tr, y_te = train_test_split(df[[x]], df[y], test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    lr = stats.linregress(df[x], df[y])
    return {"model": model, "slope": model.coef_[0], "intercept": model.intercept_,
            "p_value_slope": lr.pvalue, "r_value": lr.rvalue, "std_err": lr.stderr,
            "metrics": regression_metrics(y_te, pred), "x": x, "y": y,
            "summary": ols.summary().as_text(), "y_test": y_te.to_numpy(), "pred": pred,
            "resid": y_te.to_numpy() - pred}


def multiple_linear_regression(df, target="Wins", features=None):
    """MULTIPLE linear regression: SEVERAL predictors.  y = b0 + b1*x1 + ... + bk*xk
    Returns OLS statistics (coef, p-values, F-test, adj R2) and test-set metrics."""
    features = features or get_features(target)
    X_tr, X_te, y_tr, y_te = train_test_split(df[features], df[target], test_size=0.2,
                                              random_state=SEED)
    ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    ci = ols.conf_int()
    coefs = pd.DataFrame({"Coefficient": ols.params, "Std Err": ols.bse, "t": ols.tvalues,
                          "p-value": ols.pvalues, "CI 2.5%": ci[0], "CI 97.5%": ci[1]})
    pred = ols.predict(sm.add_constant(X_te, has_constant="add"))
    return {"ols": ols, "coefs": coefs, "features": features, "target": target,
            "r2": ols.rsquared, "adj_r2": ols.rsquared_adj, "f_stat": ols.fvalue,
            "f_pvalue": ols.f_pvalue, "metrics": regression_metrics(y_te, pred),
            "summary": ols.summary().as_text(), "y_test": y_te.to_numpy(), "pred": pred.to_numpy(),
            "resid": ols.resid.to_numpy(), "fitted": ols.fittedvalues.to_numpy()}


# ------------------------------------------------ 12-21 STATISTICAL TESTS
def _decision(p, alpha=0.05):
    return "Reject H0" if p < alpha else "Fail to reject H0"


def confidence_interval(series, conf=0.95):
    s = series.dropna()
    m, se = s.mean(), stats.sem(s)
    lo, hi = stats.t.interval(conf, len(s) - 1, loc=m, scale=se)
    return m, lo, hi


def run_statistical_tests(df, alpha=0.05, mu_starts=50, mu_wins=2, sample_n=100) -> pd.DataFrame:
    rows = []

    def add(test, h0, stat, p, extra=""):
        rows.append({"Test": test, "H0": h0, "Statistic": round(float(stat), 4),
                     "p-value": float(p), "Decision (α=%.2f)" % alpha: _decision(p, alpha),
                     "Notes": extra})

    # 13 One-sample Z-test (population sigma known = std of the full list)
    samp = draw_sample(df, sample_n)["Starts"]
    sigma = df["Starts"].std(ddof=0)
    z = (samp.mean() - mu_starts) / (sigma / np.sqrt(len(samp)))
    add("One-sample Z-test", f"mean Starts = {mu_starts}", z, 2 * (1 - stats.norm.cdf(abs(z))),
        f"n={len(samp)}, sample mean={samp.mean():.1f}, σ={sigma:.1f}")

    # 14 One-sample t-test
    t, p = stats.ttest_1samp(df["Wins"], mu_wins)
    add("One-sample t-test", f"mean Wins = {mu_wins}", t, p, f"mean={df['Wins'].mean():.2f}")

    # 16 Pearson
    r, p = stats.pearsonr(df["Starts"], df["Wins"])
    add("Pearson correlation", "ρ(Starts, Wins) = 0", r, p, f"r={r:.3f}")

    # 17 Spearman
    rho, p = stats.spearmanr(df["Starts"], df["Wins"])
    add("Spearman correlation", "ρs(Starts, Wins) = 0", rho, p, f"rho={rho:.3f}")

    exp = df[df["Starts"] >= 20]
    # 18 Two-sample independent (Welch) t-test
    modern = exp[exp["First_Season"] >= 2000]["Win_Rate"]
    classic = exp[exp["First_Season"] < 2000]["Win_Rate"]
    t, p = stats.ttest_ind(modern, classic, equal_var=False)
    add("Two-sample t-test (Welch)", "mean Win_Rate modern(2000+) = classic(<2000)", t, p,
        f"modern={modern.mean():.3f} (n={len(modern)}), classic={classic.mean():.3f} (n={len(classic)})")

    # 19 Paired t-test
    t, p = stats.ttest_rel(exp["Poles"], exp["Wins"])
    add("Paired t-test", "mean(Poles - Wins) = 0 (same driver)", t, p,
        f"mean Poles={exp['Poles'].mean():.2f}, mean Wins={exp['Wins'].mean():.2f}")

    # 20 Chi-square independence
    ct = pd.crosstab(df["Era"], df["Has_Won"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    add("Chi-square independence", "Era and Has_Won are independent", chi2, p, f"dof={dof}")

    # 21 One-way ANOVA
    groups = [g["Win_Rate"].values for _, g in exp.groupby("Era") if len(g) > 1]
    f, p = stats.f_oneway(*groups)
    add("One-way ANOVA", "mean Win_Rate equal across Eras", f, p, f"{len(groups)} groups")
    return pd.DataFrame(rows)


def cov_corr(df, cols=("Starts", "Poles", "Podiums", "Wins", "Fastest_Laps", "Championships", "Career_Years")):
    cols = list(cols)
    return df[cols].cov(), df[cols].corr(method="pearson"), df[cols].corr(method="spearman")


# ------------------------------------------- 22 MODEL COMPARISON
BASE_FEATURES = ["Starts", "Poles", "Podiums", "Fastest_Laps", "Championships", "Career_Years"]


def get_features(target="Wins"):
    return [f for f in BASE_FEATURES if f != target]


def _candidates():
    return {
        "Multiple Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"m__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso": (Lasso(max_iter=20000), {"m__alpha": [0.001, 0.01, 0.1, 1]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"m__max_depth": [3, 5, 8, None], "m__min_samples_leaf": [1, 3, 5]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                          {"m__n_estimators": [200, 400], "m__max_depth": [None, 6, 10]}),
    }


def train_models(df, target="Wins"):
    feats = get_features(target)
    X, y = df[feats], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    results, fitted, preds = [], {}, {}
    for name, (est, grid) in _candidates().items():
        pipe = Pipeline([("scale", StandardScaler()), ("m", est)])
        gs = GridSearchCV(pipe, grid or {"m__fit_intercept": [True]}, cv=5,
                          scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
        p = gs.predict(X_te)
        m = regression_metrics(y_te, p)
        m.update({"Model": name, "CV_RMSE": -gs.best_score_, "Best_Params": str(gs.best_params_)})
        results.append(m)
        fitted[name], preds[name] = gs.best_estimator_, p
    res = (pd.DataFrame(results)[["Model", "MAE", "MSE", "RMSE", "R2", "CV_RMSE", "Best_Params"]]
           .sort_values("RMSE").reset_index(drop=True))
    best = res.loc[0, "Model"]
    return {"results": res, "models": fitted, "best_name": best, "best_model": fitted[best],
            "features": feats, "target": target, "y_test": y_test_to_series(y_te), "preds": preds,
            "X_test": X_te}


def y_test_to_series(y):
    return pd.Series(np.asarray(y), name="actual")
