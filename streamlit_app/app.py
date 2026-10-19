"""
Statistical Tests Lab — Streamlit app
Load sample CSV templates or upload your own data, then run:
  Z-Test | t-Test | Chi-Square | ANOVA | Correlation | A/B Test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

SAMPLE_DIR = Path(__file__).parent / "sample_data"
ALPHA_DEFAULT = 0.05

TESTS = {
    "Z-Test (One-Sample)": {
        "file": "01_z_test_battery.csv",
        "hint": "CSV needs one numeric column (e.g. battery_hours). Set μ₀ and known σ in the sidebar.",
        "columns": "battery_hours",
    },
    "t-Test (One-Sample)": {
        "file": "02_t_test_smoothie.csv",
        "hint": "CSV needs one numeric column (e.g. quiz_score). Set hypothesized mean μ₀ in the sidebar.",
        "columns": "quiz_score",
    },
    "Chi-Square (Goodness-of-Fit)": {
        "file": "03_chi_square_candy.csv",
        "hint": "CSV needs category + observed_count columns (or one column of category labels).",
        "columns": "color, observed_count",
    },
    "ANOVA (One-Way)": {
        "file": "04_anova_study_methods.csv",
        "hint": "CSV needs a group column + a numeric value column (e.g. study_method, exam_score).",
        "columns": "study_method, exam_score",
    },
    "Correlation & Regression": {
        "file": "05_correlation_study_hours.csv",
        "hint": "CSV needs two numeric columns (e.g. study_hours, exam_score).",
        "columns": "study_hours, exam_score",
    },
    "A/B Test (Two Proportions)": {
        "file": "06_ab_test_button.csv",
        "hint": "CSV needs variant + converted (0/1), or variant + successes + trials.",
        "columns": "variant, converted",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample(filename: str) -> pd.DataFrame:
    path = SAMPLE_DIR / filename
    return pd.read_csv(path)


def verdict_box(reject: bool, detail: str):
    if reject:
        st.success(f"**Decision: REJECT H₀**  \n{detail}")
    else:
        st.info(f"**Decision: FAIL TO REJECT H₀**  \n{detail}")


def metric_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------
def run_z_test(df: pd.DataFrame, value_col: str, mu0: float, sigma: float, alpha: float):
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    n = len(x)
    xbar = float(x.mean())
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se
    z_crit = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    reject = abs(z) > z_crit

    st.subheader("One-Sample Z-Test Results")
    metric_row([
        ("n", str(n)),
        ("Mean", f"{xbar:.3f}"),
        ("Z", f"{z:+.3f}"),
        ("p-value", f"{p:.4g}"),
        ("Critical |Z|", f"{z_crit:.3f}"),
    ])
    verdict_box(reject, f"|Z|={abs(z):.3f} vs critical {z_crit:.3f} at α={alpha}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    xs = np.linspace(-4.5, 4.5, 500)
    ax.plot(xs, stats.norm.pdf(xs), color="#38bdf8", lw=2)
    ax.fill_between(xs, 0, stats.norm.pdf(xs), where=(xs <= -z_crit) | (xs >= z_crit), color="#ef4444", alpha=0.4)
    ax.axvline(z, color="#f59e0b", lw=2.5, label=f"Z={z:+.2f}")
    ax.axvline(-z_crit, color="#f87171", ls="--")
    ax.axvline(z_crit, color="#f87171", ls="--")
    ax.set_title("Standard Normal — Rejection Regions")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Python snippet"):
        st.code(
            f"""import numpy as np
from scipy import stats
x = df['{value_col}']
n, xbar, sigma, mu0 = {n}, {xbar:.4f}, {sigma}, {mu0}
se = sigma / np.sqrt(n)
z = (xbar - mu0) / se
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(z, p)""",
            language="python",
        )


def run_t_test(df: pd.DataFrame, value_col: str, mu0: float, alpha: float):
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    n = len(x)
    df_ = n - 1
    xbar = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)
    t_stat, p = stats.ttest_1samp(x, popmean=mu0)
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_)
    reject = abs(t_stat) > t_crit

    st.subheader("One-Sample t-Test Results")
    metric_row([
        ("n / df", f"{n} / {df_}"),
        ("Mean", f"{xbar:.3f}"),
        ("s", f"{s:.3f}"),
        ("t", f"{t_stat:+.3f}"),
        ("p-value", f"{p:.4g}"),
    ])
    verdict_box(reject, f"|t|={abs(t_stat):.3f} vs critical {t_crit:.3f} (df={df_})")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    xs = np.linspace(-5, 5, 500)
    ax.plot(xs, stats.t.pdf(xs, df_), color="#a855f7", lw=2)
    ax.fill_between(xs, 0, stats.t.pdf(xs, df_), where=(xs <= -t_crit) | (xs >= t_crit), color="#ef4444", alpha=0.4)
    ax.axvline(t_stat, color="#f59e0b", lw=2.5, label=f"t={t_stat:+.2f}")
    ax.set_title(f"Student's t (df={df_})")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Python snippet"):
        st.code(
            f"""from scipy import stats
t, p = stats.ttest_1samp(df['{value_col}'], popmean={mu0})
print(t, p)""",
            language="python",
        )


def run_chi_square(df: pd.DataFrame, cat_col: str, count_col: str | None, alpha: float):
    if count_col and count_col in df.columns:
        observed = pd.to_numeric(df[count_col], errors="coerce").dropna().values
        labels = df[cat_col].astype(str).tolist()
    else:
        vc = df[cat_col].value_counts().sort_index()
        labels = vc.index.astype(str).tolist()
        observed = vc.values.astype(float)

    k = len(observed)
    expected = np.full(k, observed.sum() / k)
    chi2, p = stats.chisquare(observed, expected)
    df_chi = k - 1
    chi_crit = stats.chi2.ppf(1 - alpha, df_chi)
    reject = chi2 > chi_crit

    st.subheader("Chi-Square Goodness-of-Fit Results")
    metric_row([
        ("Categories", str(k)),
        ("χ²", f"{chi2:.3f}"),
        ("Critical", f"{chi_crit:.3f}"),
        ("df", str(df_chi)),
        ("p-value", f"{p:.4g}"),
    ])
    verdict_box(reject, f"χ²={chi2:.3f} vs critical {chi_crit:.3f}")

    out = pd.DataFrame({"category": labels, "observed": observed, "expected": expected})
    st.dataframe(out, use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, observed, color="#38bdf8", label="Observed")
    axes[0].bar(labels, expected, color="#64748b", alpha=0.5, label="Expected")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    axes[0].set_title("Observed vs Expected")

    xs = np.linspace(0.01, max(chi_crit * 2.2, chi2 * 1.4, 12), 400)
    axes[1].plot(xs, stats.chi2.pdf(xs, df_chi), color="#f59e0b", lw=2)
    axes[1].axvline(chi2, color="#38bdf8", lw=2.5, label=f"χ²={chi2:.2f}")
    axes[1].axvline(chi_crit, color="#ef4444", ls="--", label=f"crit={chi_crit:.2f}")
    axes[1].legend()
    axes[1].set_title(f"χ² distribution (df={df_chi})")
    st.pyplot(fig)
    plt.close(fig)


def run_anova(df: pd.DataFrame, group_col: str, value_col: str, alpha: float):
    groups = [g[value_col].dropna().astype(float).values for _, g in df.groupby(group_col)]
    labels = [str(name) for name, _ in df.groupby(group_col)]
    f_stat, p = stats.f_oneway(*groups)
    k = len(groups)
    n = sum(len(g) for g in groups)
    df1, df2 = k - 1, n - k
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    reject = f_stat > f_crit

    st.subheader("One-Way ANOVA Results")
    metric_row([
        ("Groups", str(k)),
        ("N", str(n)),
        ("F", f"{f_stat:.3f}"),
        ("Critical F", f"{f_crit:.3f}"),
        ("p-value", f"{p:.4g}"),
    ])
    verdict_box(reject, f"F={f_stat:.3f} vs critical {f_crit:.3f}")

    summary = df.groupby(group_col)[value_col].agg(["count", "mean", "std"]).round(3)
    st.dataframe(summary, use_container_width=True)

    if HAS_STATSMODELS and reject:
        tukey = pairwise_tukeyhsd(df[value_col].astype(float), df[group_col].astype(str), alpha=alpha)
        st.write("**Tukey HSD post-hoc**")
        st.text(str(tukey))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].boxplot(groups, patch_artist=True)
    axes[0].set_xticks(range(1, len(labels) + 1))
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_title("Scores by group")
    xs = np.linspace(0.01, max(f_crit * 3, f_stat * 1.3, 8), 400)
    axes[1].plot(xs, stats.f.pdf(xs, df1, df2), color="#10b981", lw=2)
    axes[1].axvline(f_stat, color="#f59e0b", lw=2.5, label=f"F={f_stat:.2f}")
    axes[1].axvline(f_crit, color="#ef4444", ls="--", label=f"crit={f_crit:.2f}")
    axes[1].legend()
    axes[1].set_title(f"F distribution (df1={df1}, df2={df2})")
    st.pyplot(fig)
    plt.close(fig)


def run_correlation(df: pd.DataFrame, x_col: str, y_col: str, alpha: float):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    r, p = stats.pearsonr(x, y)
    slope, intercept, _, _, se = stats.linregress(x, y)
    reject = p < alpha

    st.subheader("Pearson Correlation & OLS Regression")
    metric_row([
        ("n", str(len(x))),
        ("r", f"{r:+.4f}"),
        ("R²", f"{r**2*100:.1f}%"),
        ("p-value", f"{p:.4g}"),
        ("Slope", f"{slope:.3f}"),
    ])
    verdict_box(reject, f"H₀: ρ=0. Model: {y_col} ≈ {intercept:.2f} + {slope:.2f}×{x_col}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, alpha=0.8, color="#38bdf8", edgecolors="white")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, intercept + slope * xs, color="#f59e0b", lw=2.5,
            label=f"y = {intercept:.2f} + {slope:.2f}x")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.set_title(f"Scatter + Fit (r={r:+.3f})")
    st.pyplot(fig)
    plt.close(fig)


def run_ab_test(df: pd.DataFrame, variant_col: str, converted_col: str | None,
                successes_col: str | None, trials_col: str | None, alpha: float):
    if converted_col and converted_col in df.columns:
        summary = df.groupby(variant_col)[converted_col].agg(["sum", "count"])
        summary.columns = ["successes", "trials"]
    else:
        summary = df[[variant_col, successes_col, trials_col]].copy()
        summary = summary.set_index(variant_col)
        summary.columns = ["successes", "trials"]

    if len(summary) != 2:
        st.error("A/B test needs exactly 2 variants in the data.")
        st.dataframe(summary)
        return

    labels = list(summary.index.astype(str))
    s_a, n_a = int(summary.iloc[0]["successes"]), int(summary.iloc[0]["trials"])
    s_b, n_b = int(summary.iloc[1]["successes"]), int(summary.iloc[1]["trials"])
    p_a, p_b = s_a / n_a, s_b / n_b
    lift = p_b - p_a
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    reject = abs(z) > z_crit

    st.subheader("A/B Test — Two-Proportion Z-Test")
    metric_row([
        (f"{labels[0]} rate", f"{p_a*100:.2f}%"),
        (f"{labels[1]} rate", f"{p_b*100:.2f}%"),
        ("Lift", f"{lift*100:+.2f} pp"),
        ("Z", f"{z:+.3f}"),
        ("p-value", f"{p_val:.4g}"),
    ])
    verdict_box(reject, f"|Z|={abs(z):.3f} vs critical {z_crit:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, [p_a * 100, p_b * 100], color=["#22c55e", "#f97316"])
    axes[0].set_ylabel("Conversion %")
    axes[0].set_title("Conversion by variant")
    xs = np.linspace(-4.5, 4.5, 500)
    axes[1].plot(xs, stats.norm.pdf(xs), color="#38bdf8", lw=2)
    axes[1].axvline(z, color="#f59e0b", lw=2.5, label=f"Z={z:+.2f}")
    axes[1].axvline(-z_crit, color="#ef4444", ls="--")
    axes[1].axvline(z_crit, color="#ef4444", ls="--")
    axes[1].legend()
    axes[1].set_title("Two-proportion Z distribution")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# App UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Tests Lab",
    page_icon="📊",
    layout="wide",
)

st.title("Statistical Tests Lab")
st.caption("Load a sample CSV template or upload your own data — then run Z, t, Chi-Square, ANOVA, Correlation, or A/B tests.")

with st.sidebar:
    st.header("1. Choose test")
    test_name = st.selectbox("Test", list(TESTS.keys()))
    meta = TESTS[test_name]
    st.info(meta["hint"])

    st.header("2. Load data")
    source = st.radio("Data source", ["Use sample template CSV", "Upload my CSV"], index=0)

    df = None
    if source == "Use sample template CSV":
        sample_path = SAMPLE_DIR / meta["file"]
        if sample_path.exists():
            df = load_sample(meta["file"])
            st.success(f"Loaded `{meta['file']}`")
            st.download_button(
                "Download this template CSV",
                data=sample_path.read_bytes(),
                file_name=meta["file"],
                mime="text/csv",
            )
        else:
            st.error(f"Missing sample file: {meta['file']}")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success(f"Uploaded `{uploaded.name}`")

    alpha = st.slider("Significance level α", 0.01, 0.10, ALPHA_DEFAULT, 0.01)

    st.divider()
    st.markdown("**Expected columns**")
    st.code(meta["columns"])

if df is None:
    st.warning("Load a sample template or upload a CSV to continue.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

st.subheader("3. Map columns & run")

if test_name.startswith("Z-Test"):
    c1, c2, c3 = st.columns(3)
    value_col = c1.selectbox("Value column", numeric_cols or all_cols)
    mu0 = c2.number_input("Hypothesized mean μ₀", value=24.0)
    sigma = c3.number_input("Known population σ", value=2.0, min_value=0.0001)
    if st.button("Run Z-Test", type="primary"):
        run_z_test(df, value_col, mu0, sigma, alpha)

elif test_name.startswith("t-Test"):
    c1, c2 = st.columns(2)
    value_col = c1.selectbox("Value column", numeric_cols or all_cols)
    mu0 = c2.number_input("Hypothesized mean μ₀", value=70.0)
    if st.button("Run t-Test", type="primary"):
        run_t_test(df, value_col, mu0, alpha)

elif test_name.startswith("Chi-Square"):
    c1, c2 = st.columns(2)
    cat_col = c1.selectbox("Category column", all_cols)
    count_options = ["(auto-count from category labels)"] + numeric_cols
    count_choice = c2.selectbox("Count column (optional)", count_options)
    count_col = None if count_choice.startswith("(auto") else count_choice
    if st.button("Run Chi-Square", type="primary"):
        run_chi_square(df, cat_col, count_col, alpha)

elif test_name.startswith("ANOVA"):
    c1, c2 = st.columns(2)
    group_col = c1.selectbox("Group column", all_cols)
    value_col = c2.selectbox("Value column", numeric_cols or all_cols)
    if st.button("Run ANOVA", type="primary"):
        run_anova(df, group_col, value_col, alpha)

elif test_name.startswith("Correlation"):
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X column", numeric_cols or all_cols, index=0)
    y_default = 1 if len(numeric_cols) > 1 else 0
    y_col = c2.selectbox("Y column", numeric_cols or all_cols, index=y_default)
    if st.button("Run Correlation", type="primary"):
        run_correlation(df, x_col, y_col, alpha)

elif test_name.startswith("A/B"):
    mode = st.radio("A/B CSV format", ["variant + converted (0/1)", "variant + successes + trials"])
    if mode.startswith("variant + converted"):
        c1, c2 = st.columns(2)
        variant_col = c1.selectbox("Variant column", all_cols)
        converted_col = c2.selectbox("Converted column (0/1)", numeric_cols or all_cols)
        if st.button("Run A/B Test", type="primary"):
            run_ab_test(df, variant_col, converted_col, None, None, alpha)
    else:
        c1, c2, c3 = st.columns(3)
        variant_col = c1.selectbox("Variant column", all_cols)
        successes_col = c2.selectbox("Successes column", numeric_cols or all_cols)
        trials_col = c3.selectbox("Trials column", numeric_cols or all_cols)
        if st.button("Run A/B Test", type="primary"):
            run_ab_test(df, variant_col, None, successes_col, trials_col, alpha)

st.divider()
st.markdown("**Sample templates available**")
for name, info in TESTS.items():
    st.write(f"- `{info['file']}` — {name}")
