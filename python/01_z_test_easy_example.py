"""
=============================================================================
TEST 1: ONE-SAMPLE Z-TEST (EASY EVERYDAY EXAMPLE)
Topic: Smartphone Battery Life Audit (OmniPhone 24.0h Claim vs Reality)
=============================================================================
Scenario:
    A popular smartphone maker "OmniPhone" advertises:
    "Average battery life lasts 24.0 hours on a single charge."
    A tech review laboratory tests n = 64 retail phones in automated drain tests.
    The sample average battery life is X_bar = 22.8 hours (-1.2h below claim).
    The factory population standard deviation is known to be sigma = 2.0 hours.

Question:
    Is this -1.2 hour gap just normal small sampling variation, or is the
    manufacturer statistically proven to be under-delivering?
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


def run_smartphone_battery_z_test():
    print("=" * 75)
    print("TEST 1: ONE-SAMPLE Z-TEST — SMARTPHONE BATTERY LIFE AUDIT")
    print("=" * 75)

    # 1. Hypotheses & Known Parameters
    mu_0 = 24.0       # Advertised benchmark battery life (hours)
    sigma = 2.0       # Known population standard deviation (hours)
    n = 64            # Sample size of tested retail phones
    x_bar = 22.8      # Sample mean battery life (hours)
    alpha = 0.05      # Significance level

    print(f"Stated Claim (mu_0)     : {mu_0:.1f} hours")
    print(f"Known Pop Std (sigma)   : {sigma:.1f} hours")
    print(f"Tested Phones (n)       : {n} units")
    print(f"Sample Average (X_bar)  : {x_bar:.1f} hours (Deficit: {x_bar - mu_0:.1f} hrs)")
    print(f"Significance Level (a)  : {alpha}")
    print("-" * 75)

    # 2. Mathematical Calculations
    se = sigma / np.sqrt(n)                    # Standard Error: 2.0 / 8 = 0.25h
    z_stat = (x_bar - mu_0) / se               # Z = (22.8 - 24.0) / 0.25 = -4.80
    z_crit = stats.norm.ppf(1 - alpha / 2)     # +/- 1.96
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% Confidence Interval for True Mean
    ci_lower = x_bar - z_crit * se
    ci_upper = x_bar + z_crit * se

    print(f"Standard Error (SE)     : {se:.3f} hours")
    print(f"Calculated Z-Statistic  : {z_stat:+.3f}")
    print(f"Critical Z Thresholds   : +/- {z_crit:.3f}")
    print(f"Two-Tailed p-value      : {p_value:.8f} ({p_value:.2e})")
    print(f"95% Confidence Interval : [{ci_lower:.2f} hrs, {ci_upper:.2f} hrs]")
    print("-" * 75)

    # 3. Statistical Decision & Practical Takeaway
    if abs(z_stat) > z_crit and p_value < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0)"
        takeaway = (
            "STATISTICALLY SIGNIFICANT DEFICIT DETECTED!\n"
            "The probability of observing 22.8 hours or less by pure chance is < 1 in 600,000.\n"
            "OmniPhone's 24-hour marketing claim is objectively inaccurate.\n"
            "Action: Tech reviewer publishes findings; OmniPhone must issue a battery-saving firmware patch."
        )
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = "Battery life is within normal random manufacturing variation."

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("=" * 75)

    # 4. Generate Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#090d16')
    ax.set_facecolor('#0f172a')

    x = np.linspace(-5.5, 5.5, 1000)
    y = stats.norm.pdf(x, 0, 1)
    ax.plot(x, y, color='#38bdf8', lw=2.5, label='Standard Normal N(0, 1)')

    # Rejection tails
    ax.fill_between(x, 0, y, where=(x <= -z_crit), color='#ef4444', alpha=0.5, label=f'Rejection Tail (a/2 = 0.025)')
    ax.fill_between(x, 0, y, where=(x >= z_crit), color='#ef4444', alpha=0.5)
    ax.fill_between(x, 0, y, where=((x > -z_crit) & (x < z_crit)), color='#10b981', alpha=0.15, label='Acceptance Zone (95%)')

    # Lines
    ax.axvline(-z_crit, color='#f87171', ls='--', lw=1.5, label=f'Critical Z = -{z_crit:.2f}')
    ax.axvline(z_crit, color='#f87171', ls='--', lw=1.5)
    ax.axvline(z_stat, color='#f59e0b', lw=3, label=f'Audited Z = {z_stat:.2f}')

    ax.set_title('OmniPhone Battery Life: One-Sample Z-Test Distribution', color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Standard Deviations from Claimed Mean (Z)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Probability Density', color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.grid(color='#334155', alpha=0.4, ls=':')
    ax.legend(facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#f8fafc', loc='upper right')

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '01_z_test_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "One-Sample Z-Test",
        "example": "Smartphone Battery Claim Audit",
        "z_stat": z_stat,
        "p_value": p_value,
        "decision": verdict
    }


if __name__ == '__main__':
    run_smartphone_battery_z_test()
