"""
=============================================================================
TEST 6: A/B TESTING (EASY EVERYDAY EXAMPLE)
Topic: Green "Buy Now" Button vs Orange "Buy Now" Button (Conversion Rate)
=============================================================================
Scenario:
    An online store wants more purchases. Designers propose two CTA buttons:
        - Variant A (Control): Green "Buy Now" button
        - Variant B (Treatment): Orange "Buy Now" button
    They run a randomized A/B test for one week:
        - Variant A shown to n_A = 2,000 visitors → 160 purchases (8.0%)
        - Variant B shown to n_B = 2,000 visitors → 210 purchases (10.5%)

Question:
    Is the +2.5 percentage-point lift from Orange real, or just random traffic
    noise? Use a Two-Proportion Z-Test to decide.
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


def run_button_color_ab_test():
    print("=" * 75)
    print("TEST 6: A/B TESTING — GREEN vs ORANGE 'BUY NOW' BUTTON")
    print("=" * 75)

    # 1. Observed conversion counts
    n_a = 2000          # Visitors seeing Green (Control)
    n_b = 2000          # Visitors seeing Orange (Treatment)
    conversions_a = 160  # Purchases with Green
    conversions_b = 210  # Purchases with Orange
    alpha = 0.05

    p_a = conversions_a / n_a   # 0.080
    p_b = conversions_b / n_b   # 0.105
    lift = p_b - p_a            # +0.025 (2.5 percentage points)
    relative_lift = lift / p_a  # +31.25% relative uplift

    print(f"Variant A (Green Control)   : {conversions_a}/{n_a} = {p_a*100:.1f}% conversion")
    print(f"Variant B (Orange Treatment): {conversions_b}/{n_b} = {p_b*100:.1f}% conversion")
    print(f"Absolute Lift (B - A)       : {lift*100:+.1f} percentage points")
    print(f"Relative Lift               : {relative_lift*100:+.1f}%")
    print(f"Significance Level (alpha)  : {alpha}")
    print("-" * 75)

    # 2. Two-Proportion Z-Test (pooled)
    # H0: p_A = p_B   |   H1: p_A != p_B
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_b - p_a) / se
    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% CI for difference of proportions
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = lift - z_crit * se_unpooled
    ci_high = lift + z_crit * se_unpooled

    print(f"Pooled Conversion Rate      : {p_pool*100:.2f}%")
    print(f"Standard Error (SE)         : {se:.5f}")
    print(f"Z-Statistic                 : {z_stat:+.3f}")
    print(f"Critical Z Thresholds       : +/- {z_crit:.3f}")
    print(f"Two-Tailed p-value          : {p_value:.6f} ({p_value:.2e})")
    print(f"95% CI for Lift (B - A)     : [{ci_low*100:+.2f} pp, {ci_high*100:+.2f} pp]")
    print("-" * 75)

    # 3. Decision
    if abs(z_stat) > z_crit and p_value < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0)"
        takeaway = (
            "ORANGE BUTTON WINS — LIFT IS REAL!\n"
            f"Chance of seeing a +{lift*100:.1f}pp gap by pure luck: only {p_value*100:.3f}%.\n"
            "Action: Ship Orange 'Buy Now' to 100% of traffic and keep monitoring."
        )
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = "No significant difference. Keep Green (or keep testing)."

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("=" * 75)

    # 4. Distribution + bar chart visual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='#090d16')

    # Conversion rate bars
    ax1.set_facecolor('#0f172a')
    bars = ax1.bar(['A: Green\n(Control)', 'B: Orange\n(Treatment)'],
                   [p_a * 100, p_b * 100],
                   color=['#22c55e', '#f97316'], edgecolor='#ffffff', lw=1.5, width=0.55)
    ax1.bar_label(bars, fmt='%.1f%%', color='#f8fafc', fontsize=13, fontweight='bold', padding=4)
    ax1.set_ylim(0, 14)
    ax1.set_ylabel('Conversion Rate (%)', color='#94a3b8', fontsize=11)
    ax1.set_title(f'A/B Conversion Rates  (Lift = {lift*100:+.1f} pp)', color='#f8fafc',
                  fontsize=12, fontweight='bold')
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(color='#334155', alpha=0.3, ls=':', axis='y')

    # Z distribution
    ax2.set_facecolor('#0f172a')
    x = np.linspace(-4.5, 4.5, 1000)
    y = stats.norm.pdf(x)
    ax2.plot(x, y, color='#38bdf8', lw=2.5, label='Standard Normal N(0,1)')
    ax2.fill_between(x, 0, y, where=(x <= -z_crit), color='#ef4444', alpha=0.5, label='Rejection tails')
    ax2.fill_between(x, 0, y, where=(x >= z_crit), color='#ef4444', alpha=0.5)
    ax2.fill_between(x, 0, y, where=((x > -z_crit) & (x < z_crit)), color='#10b981', alpha=0.12, label='Acceptance zone')
    ax2.axvline(-z_crit, color='#f87171', ls='--', lw=1.5)
    ax2.axvline(z_crit, color='#f87171', ls='--', lw=1.5)
    ax2.axvline(z_stat, color='#f59e0b', lw=3, label=f'Audited Z = {z_stat:+.2f}')
    ax2.set_title('Two-Proportion Z-Test Distribution', color='#f8fafc', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Z-Statistic', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Density', color='#94a3b8', fontsize=11)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(color='#334155', alpha=0.3, ls=':')
    ax2.legend(facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#f8fafc', loc='upper left', fontsize=9)

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '06_ab_testing_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "A/B Test (Two-Proportion Z)",
        "example": "Green vs Orange Buy Button",
        "z_stat": z_stat,
        "p_value": p_value,
        "lift": lift,
        "decision": verdict
    }


if __name__ == '__main__':
    run_button_color_ab_test()
