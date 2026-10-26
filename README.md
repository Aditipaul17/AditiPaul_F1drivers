# 🏎️ Formula 1 Driver Performance Prediction & Statistical Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science and statistical computing project analyzing historical Formula 1 driver performance data scraped directly from Wikipedia. The project features exploratory data analysis, inferential statistics, formal hypothesis testing, regression analysis, machine learning model comparisons with hyperparameter tuning, and an interactive **Streamlit** dashboard.

---

## 📌 Project Overview

Formula 1 history spans over seven decades, featuring hundreds of drivers across different technological eras, scoring systems, and season lengths. This project investigates the key factors that govern a driver's success—quantified primarily through **Race Wins**, **Podiums**, and **Pole Positions**—using statistical inference and predictive modeling.

### Core Objectives
1. **Automated Data Pipeline**: Scrape, clean, preprocess, and engineer performance features from Wikipedia's comprehensive list of Formula 1 drivers.
2. **Exploratory Data Analysis (EDA)**: Uncover performance distributions, era-based variations, driver nationalities, and inter-feature correlations (Pearson & Spearman).
3. **Statistical Inference & Sampling**: Validate the Central Limit Theorem (CLT) on skewed racing metrics and calculate parametric confidence intervals.
4. **Hypothesis Testing**: Conduct classical parametric and non-parametric tests ($Z$-test, One-Sample $t$-test, Two-Sample $t$-test, Paired $t$-test, Chi-Square Independence, ANOVA).
5. **Predictive Modeling**: Compare multiple regression algorithms (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest) with cross-validation and hyperparameter tuning to accurately predict driver performance.
6. **Interactive GUI**: Deploy an intuitive, multi-tab Streamlit web application enabling real-time experimentation, interactive hypothesis testing, and custom driver performance predictions.

---

## 🗂️ Repository Structure

```plaintext
AditiPaul_F1drivers/
├── app.py                  # Streamlit interactive web application (8 analytical tabs)
├── f1_pipeline.py          # Core pipeline: web scraping, cleaning, stats & ML models
├── f1_plots.py             # Data visualization modules (distributions, heatmaps, residuals)
├── run_analysis.py         # Command-line execution script for the complete pipeline
├── f1_drivers_raw.csv      # Raw scraped dataset from Wikipedia
├── f1_drivers_clean.csv    # Cleaned and feature-engineered dataset
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files and folders ignored by Git
└── README.md               # Project documentation
```

---

## 🚀 Key Features

### 1. Data Scraping & Preprocessing (`f1_pipeline.py`)
- Automatically scrapes the table from `Wikipedia: List of Formula One drivers` using `requests` and `BeautifulSoup`.
- Strips footnote references, special characters, and handles drivers currently competing.
- Engineers key analytical metrics:
  - `Win_Rate`: Percentage of race starts resulting in a win ($\frac{\text{Wins}}{\text{Starts}} \times 100$)
  - `Podium_Rate`: Percentage of race starts resulting in a podium finish ($\frac{\text{Podiums}}{\text{Starts}} \times 100$)
  - `Pole_Rate`: Percentage of race starts starting from pole position ($\frac{\text{Poles}}{\text{Starts}} \times 100$)
  - `Points_Per_Start`: Average points scored per race start
  - `Career_Years`: Total span of active seasons
  - `Era`: Categorized into distinct racing eras (`1950-69`, `1970-89`, `1990-2009`, `2010+`)

### 2. Statistical Analysis & Hypothesis Testing
- **Sampling & Central Limit Theorem (CLT)**: Generates empirical sampling distributions across multiple sample sizes ($N = 2, 10, 30, 100$) showing convergence toward normality even for heavily right-skewed metrics like career wins.
- **Hypothesis Testing Suite**:
  - **One-Sample $Z$-Test**: Evaluates whether sample mean starts differ significantly from a historical benchmark.
  - **One-Sample $t$-Test**: Tests driver win averages against a hypothesized mean.
  - **Two-Sample Independent $t$-Test**: Compares win rates between the modern era (1990+) and the vintage era (pre-1990).
  - **Paired $t$-Test**: Compares career podium counts against career wins for podium-scoring drivers.
  - **Chi-Square Test of Independence**: Assesses association between era of debut and driver achievement tier (e.g., zero wins vs. race winners).
  - **One-Way ANOVA**: Tests for statistically significant differences in win rates across all four historical eras.

### 3. Machine Learning & Regression
- **Simple Linear Regression (SLR)**: Ordinary Least Squares (OLS) modeling individual relationships (e.g., `Starts -> Wins` or `Poles -> Wins`) with statistical significance tests and residual plots.
- **Multiple Linear Regression (MLR)**: Multi-predictor OLS using `statsmodels` reporting coefficients, standard errors, $t$-statistics, $p$-values, $R^2$, Adjusted $R^2$, and $F$-statistic.
- **Model Benchmark & Hyperparameter Tuning**:
  - Compares **Linear Regression**, **Ridge Regression**, **Lasso Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.
  - Evaluates performance using 5-fold cross-validation (`GridSearchCV`) and test set metrics ($R^2$, RMSE, MAE).
  - Computes and plots feature importances / standardized model coefficients.

### 4. Interactive Streamlit Web App (`app.py`)
Organized into 8 dedicated analytical tabs:
1. **📋 Data**: Inspect raw and cleaned data tables, view statistical summaries, and export cleaned CSV.
2. **📊 EDA**: Histograms with KDE, boxplots by debut era, top nationalities, and correlation heatmaps.
3. **🎲 Sampling & CLT**: Interactive sample size slider, CLT convergence visualizations, and 95% confidence intervals.
4. **📈 Simple LR**: Dynamic predictor/response selection, regression line fitting, residual plots, and full OLS summary table.
5. **📉 Multiple LR**: Multi-select feature inputs, coefficient tables, metric cards, and fitted vs. residual diagnostics.
6. **🧪 Hypothesis Tests**: Configurable significance level ($\alpha$) and null hypotheses with dynamic $p$-value and rejection verdicts.
7. **🏆 Model Comparison**: Metric comparison table across all 5 algorithms, actual vs. predicted plots, and feature importance bar charts.
8. **🔮 Predict**: Interactive form to input hypothetical or real driver statistics and predict expected career wins, podiums, or poles using the best-performing model.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Aditipaul17/AditiPaul_F1drivers.git
cd AditiPaul_F1drivers
```

### 2. Create and Activate a Virtual Environment
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Launch the Streamlit Web Application
To run the interactive web app locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run the Command-Line Analysis Pipeline
To execute the automated analysis, generate summary tables, and save all output plots:
```bash
python run_analysis.py
```
> **Tip**: Add `--refresh` if you want to re-scrape the latest driver data from Wikipedia:
> ```bash
> python run_analysis.py --refresh
> ```
All generated plots (`.png`), hypothesis test tables (`.csv`), model comparison metrics (`.csv`), and the serialized best model (`.joblib`) will be saved in an `outputs/` directory.

---

## 📊 Technologies Used

| Category | Libraries / Tools |
| :--- | :--- |
| **Language** | Python 3 |
| **Web App** | Streamlit |
| **Web Scraping** | BeautifulSoup4, Requests, lxml |
| **Data Manipulation** | Pandas, NumPy |
| **Statistical Analysis** | SciPy, Statsmodels |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Visualization** | Matplotlib, Seaborn |

---

## 👤 Author

- **Aditi Paul** - [@Aditipaul17](https://github.com/Aditipaul17)
- Email: `24paula@rbunagpur.in`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
