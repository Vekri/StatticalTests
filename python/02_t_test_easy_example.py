"""
=============================================================================
TEST 2: STUDENT'S T-TEST (EASY EVERYDAY EXAMPLE)
Topic: Berry Brain Smoothie vs Math Quiz Score (Small Sample Pilot, n = 12)
=============================================================================
Scenario:
    A high school's weekly 100-point math quiz benchmark average is mu_0 = 70.0.
    A student wellness club develops a "Berry Brain Smoothie" with matcha.
    Because it's a small experimental batch, only n = 12 students volunteer.
    The population standard deviation sigma of this new recipe is UNKNOWN.
    The 12 students drink the smoothie and achieve an average score of X_bar = 76.5
    with a sample standard deviation s = 6.2 points.

Question:
    Did the smoothie genuinely boost student quiz scores, or was this small
    group of 12 students just lucky?
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


def run_smoothie_brain_t_test():
    print("=" * 75)
    print("TEST 2: STUDENT'S T-TEST — BERRY BRAIN SMOOTHIE MATH PILOT")
    print("=" * 75)

    # 1. Historical Benchmark & Sample Data (n = 12)
    mu_0 = 70.0  # Historical quiz average
    alpha = 0.05
    scores = np.array([76.0, 81.0, 72.0, 85.0, 78.0, 70.0, 79.0, 83.0, 74.0, 80.0, 68.0, 72.0])
    n = len(scores)
    df = n - 1

    x_bar = np.mean(scores)
    s = np.std(scores, ddof=1)  # Sample standard deviation (ddof=1)
    se = s / np.sqrt(n)

    print(f"Historical Quiz Mean (mu_0) : {mu_0:.1f} pts")
    print(f"Volunteer Sample Size (n)   : {n} students (Degrees of Freedom = {df})")
    print(f"Smoothie Sample Mean (X_bar): {x_bar:.2f} pts (+{x_bar - mu_0:.2f} pts gain)")
    print(f"Sample Std Dev (s)          : {s:.2f} pts (Population sigma is UNKNOWN)")
    print(f"Standard Error (SE)         : {se:.3f} pts")
    print("-" * 75)

    # 2. Student's t-test calculation via scipy.stats
    t_stat, p_value = stats.ttest_1samp(scores, popmean=mu_0)
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)

    # 95% Confidence Interval for True Mean
    ci_lower = x_bar - t_crit * se
    ci_upper = x_bar + t_crit * se

    print(f"Calculated t-Statistic      : {t_stat:+.3f}")
    print(f"Critical t Thresholds       : +/- {t_crit:.3f} (df = {df})")
    print(f"Two-Tailed p-value          : {p_value:.5f}")
    print(f"95% Confidence Interval     : [{ci_lower:.2f} pts, {ci_upper:.2f} pts]")
    print("-" * 75)

    # 3. Statistical Decision & Practical Takeaway
    if abs(t_stat) > t_crit and p_value < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0)"
        takeaway = (
            "STATISTICALLY SIGNIFICANT COGNITIVE BOOST CONFIRMED!\n"
            f"The probability of observing this +{x_bar - mu_0:.1f} pts boost by chance is only {p_value*100:.2f}%.\n"
            "Even with a small sample of 12 students, the effect is robust and positive.\n"
            "Action: The school approves serving Berry Brain Smoothies before morning exams!"
        )
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = "Differences are within normal small-sample test variance."

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("=" * 75)

    # 4. Generate Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#090d16')
    ax.set_facecolor('#0f172a')

    x = np.linspace(-5, 5, 1000)
    y = stats.t.pdf(x, df=df)
    ax.plot(x, y, color='#a855f7', lw=2.5, label=f"Student's t-Distribution (df={df})")

    # Rejection tails
    ax.fill_between(x, 0, y, where=(x <= -t_crit), color='#ef4444', alpha=0.5, label='Rejection Tail (a/2 = 0.025)')
    ax.fill_between(x, 0, y, where=(x >= t_crit), color='#ef4444', alpha=0.5)
    ax.fill_between(x, 0, y, where=((x > -t_crit) & (x < t_crit)), color='#10b981', alpha=0.15, label='Acceptance Zone (95%)')

    # Lines
    ax.axvline(-t_crit, color='#f87171', ls='--', lw=1.5, label=f'Critical t = -{t_crit:.2f}')
    ax.axvline(t_crit, color='#f87171', ls='--', lw=1.5, label=f'Critical t = +{t_crit:.2f}')
    ax.axvline(t_stat, color='#f59e0b', lw=3, label=f'Audited t = {t_stat:.2f}')

    ax.set_title(f"Berry Smoothie Quiz Test: Student's t-Distribution (df={df})", color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-Statistic Value', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Probability Density', color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.grid(color='#334155', alpha=0.4, ls=':')
    ax.legend(facecolor='#1e293b', edgecolor='#a855f7', labelcolor='#f8fafc', loc='upper right')

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '02_t_test_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "Student's t-Test",
        "example": "Berry Brain Smoothie Pilot",
        "t_stat": t_stat,
        "p_value": p_value,
        "decision": verdict
    }


if __name__ == '__main__':
    run_smoothie_brain_t_test()
