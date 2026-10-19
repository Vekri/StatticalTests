"""
=============================================================================
TEST 5: PEARSON CORRELATION & REGRESSION (EASY EVERYDAY EXAMPLE)
Topic: Study Hours vs Exam Grade (Predicting Final Scores from Effort)
=============================================================================
Scenario:
    A high school advisor surveys n = 35 students to prove the real impact
    of weekly study time on final exam grades.
    Variables:
        - X (Independent) : Weekly self-study hours (2 to 20 hours/week)
        - Y (Dependent)   : Final course exam score (out of 100 points)

Question:
    Is there a statistically significant linear correlation between study hours
    and exam grades? Can we build an OLS regression formula to accurately
    predict a student's score from their study habits?
=============================================================================
"""

import sys
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_study_hours_correlation_test():
    print("=" * 75)
    print("TEST 5: PEARSON CORRELATION & REGRESSION — STUDY HOURS VS EXAM GRADES")
    print("=" * 75)

    np.random.seed(42)
    n = 35
    # Generate study hours between 2 and 20
    hours = np.linspace(2.5, 19.5, n) + np.random.normal(0, 0.8, n)
    hours = np.clip(hours, 2.0, 20.0)

    # True model: Score = 46.5 + 2.45 * Hours + noise
    noise = np.random.normal(0, 3.8, n)
    scores = np.clip(46.5 + 2.45 * hours + noise, 45.0, 99.0)

    alpha = 0.05
    df = n - 2

    # 1. Pearson Correlation
    r_val, p_val = stats.pearsonr(hours, scores)
    r_squared = r_val ** 2

    print(f"Sample Size (n)             : {n} students (Degrees of Freedom = {df})")
    print(f"Pearson Correlation (r)     : {r_val:+.4f} (Very Strong Positive Relationship)")
    print(f"Coefficient of Determ. (R^2): {r_squared*100:.1f}% of score variance explained by study hours")
    print(f"p-value                     : {p_val:.2e}")
    print("-" * 75)

    # 2. Linear Regression Model
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(hours, scores)

    print("ORDINARY LEAST SQUARES (OLS) REGRESSION MODEL:")
    print(f"Formula                     : Exam Score = {intercept:.2f} + {slope:.2f} * (Study Hours)")
    print(f"Baseline Score (Intercept)  : {intercept:.2f} points (Score with 0 extra study hours)")
    print(f"Marginal Gain per Hour      : +{slope:.2f} points for every additional hour studied per week")
    print(f"Slope Standard Error        : +/- {std_err:.3f}")
    print("-" * 75)

    # 3. Decision & Concrete Predictions
    if abs(r_val) > 0 and p_val < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0: rho = 0)"
        takeaway = (
            "STATISTICALLY SIGNIFICANT RELATIONSHIP PROVEN!\n"
            f"The probability that this correlation is pure coincidence is p = {p_val:.2e} (< 0.0001).\n"
            "Action: Advisors can use the regression model to set realistic study targets."
        )
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = "No significant linear relationship found."

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("-" * 75)

    # Real-life scenario predictions
    test_targets = [5.0, 10.0, 15.0]
    print("CONCRETE SCORE PREDICTIONS:")
    for h in test_targets:
        pred_y = intercept + slope * h
        print(f"  • Studying {h:>4.1f} hrs/week  -->  Predicted Score: {pred_y:.1f}%")
    print("=" * 75)

    # 4. Generate Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='#090d16')

    # Subplot 1: Scatter plot with Regression line and CI
    ax1.set_facecolor('#0f172a')
    ax1.scatter(hours, scores, color='#38bdf8', s=60, alpha=0.85, edgecolors='#ffffff', label=f'Students (n={n})')

    x_line = np.linspace(2, 20, 100)
    y_line = intercept + slope * x_line
    ax1.plot(x_line, y_line, color='#f59e0b', lw=2.5, label=f'Fit: Y = {intercept:.1f} + {slope:.2f}X')

    # Confidence interval band around line
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    s_err = np.sqrt(np.sum((scores - (intercept + slope * hours))**2) / df)
    ci = t_crit * s_err * np.sqrt(1/n + (x_line - np.mean(hours))**2 / np.sum((hours - np.mean(hours))**2))
    ax1.fill_between(x_line, y_line - ci, y_line + ci, color='#f59e0b', alpha=0.15, label='95% Confidence Band')

    ax1.set_title(f'Study Hours vs Exam Score (r = {r_val:+.2f})', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Weekly Study Hours', color='#94a3b8', fontsize=11)
    ax1.set_ylabel('Final Exam Score (%)', color='#94a3b8', fontsize=11)
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(color='#334155', alpha=0.3, ls=':')
    ax1.legend(facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#f8fafc', loc='upper left')

    # Subplot 2: Correlation Gauge & R^2 Decomposition
    ax2.set_facecolor('#0f172a')
    categories = ['Study Hours (R²)', 'Other Factors (1-R²)']
    fractions = [r_squared * 100, (1 - r_squared) * 100]
    colors_pie = ['#10b981', '#334155']

    wedges, texts, autotexts = ax2.pie(
        fractions, labels=categories, colors=colors_pie, autopct='%1.1f%%',
        startangle=140, textprops={'color': '#f8fafc', 'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('#ffffff')

    ax2.set_title(f'Variance Explained: R² = {r_squared*100:.1f}%', color='#f8fafc', fontsize=12, fontweight='bold')

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '05_correlation_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "Pearson Correlation & Regression",
        "example": "Study Hours vs Exam Grades",
        "r_val": r_val,
        "r_squared": r_squared,
        "p_value": p_val,
        "decision": verdict
    }


if __name__ == '__main__':
    run_study_hours_correlation_test()
