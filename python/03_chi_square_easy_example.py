"""
=============================================================================
TEST 3: CHI-SQUARE TEST (EASY EVERYDAY EXAMPLE)
Topic: M&M / Skittles Candy Bag Color Audit (Observed vs Expected 20% Each)
=============================================================================
Scenario:
    A candy manufacturer claims:
    "Every bag contains an equal 20% mix of all 5 fruity colors:
     Red, Blue, Green, Yellow, and Orange."
    A student opens a jumbo bag of N = 100 candies and sorts them into bowls.
    Expected count: E = 20 candies for each color (since 100 / 5 = 20).
    Observed counts:
        Red: 15, Blue: 18, Green: 28, Yellow: 14, Orange: 25.
    At first glance, Green (28) seems too high and Yellow (14) seems too low!

Question:
    Does this prove the factory packaging machine is biased, or is this
    uneven bowl just natural random candy bagging variance?
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


def run_candy_color_chi_square_test():
    print("=" * 75)
    print("TEST 3: CHI-SQUARE GOODNESS-OF-FIT — CANDY BAG COLOR AUDIT")
    print("=" * 75)

    # 1. Categories & Frequencies
    colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange']
    observed = np.array([15, 18, 28, 14, 25])
    total_n = np.sum(observed)  # 100 candies
    k = len(colors)             # 5 categories
    df = k - 1                  # 4 degrees of freedom
    alpha = 0.05

    # Expected frequency under H0 (equal 20% distribution)
    expected = np.array([total_n / k] * k)  # 20 each

    print(f"Total Candies Sampled (N)  : {total_n}")
    print(f"Color Categories (k)       : {k} colors (Degrees of Freedom = {df})")
    print(f"Expected per Color (E)     : {expected[0]:.1f} candies (Assumption E >= 5: MET)")
    print("-" * 75)
    print(f"{'Color':<10} | {'Observed (O)':<14} | {'Expected (E)':<14} | {'(O - E)^2 / E':<15}")
    print("-" * 75)

    terms = []
    for c, o, e in zip(colors, observed, expected):
        term = ((o - e) ** 2) / e
        terms.append(term)
        print(f"{c:<10} | {o:<14} | {e:<14.1f} | {term:<15.2f}")

    manual_chi2 = np.sum(terms)
    print("-" * 75)

    # 2. Scipy Chi-Square Test
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    chi2_crit = stats.chi2.ppf(1 - alpha, df=df)

    print(f"Calculated Chi-Square (X^2) : {chi2_stat:.3f}")
    print(f"Critical Chi-Square Threshold: {chi2_crit:.3f} (alpha = {alpha}, df = {df})")
    print(f"p-value                     : {p_value:.5f}")
    print("-" * 75)

    # 3. Statistical Decision & Practical Takeaway
    if chi2_stat > chi2_crit and p_value < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0)"
        takeaway = "Factory machine is significantly biased toward certain colors."
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = (
            "THE CANDY FACTORY IS HONEST!\n"
            f"Even though Green (28) and Yellow (14) look uneven to the human eye,\n"
            f"the Chi-Square value of {chi2_stat:.2f} is well below the critical threshold of {chi2_crit:.2f}.\n"
            f"A deviation of this size occurs {p_value*100:.1f}% of the time purely by natural random chance.\n"
            "Action: No factory recall or consumer complaint needed. The mix is fair!"
        )

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("=" * 75)

    # 4. Generate Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='#090d16')

    # Subplot 1: Bar chart comparing Observed vs Expected
    ax1.set_facecolor('#0f172a')
    x_idx = np.arange(k)
    bar_width = 0.35
    c_palette = ['#ef4444', '#3b82f6', '#10b981', '#eab308', '#f97316']

    ax1.bar(x_idx - bar_width/2, observed, bar_width, label='Observed (O)', color=c_palette, edgecolor='#ffffff', lw=1.2)
    ax1.bar(x_idx + bar_width/2, expected, bar_width, label='Expected (E = 20)', color='#475569', alpha=0.7, edgecolor='#94a3b8', hatch='//')

    ax1.set_title('Candies: Observed vs Expected Mix', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(colors, color='#f8fafc')
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(color='#334155', alpha=0.3, ls=':', axis='y')
    ax1.legend(facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#f8fafc', loc='upper right')

    # Subplot 2: Chi-Square Distribution curve
    ax2.set_facecolor('#0f172a')
    x = np.linspace(0.01, 20, 1000)
    y = stats.chi2.pdf(x, df=df)
    ax2.plot(x, y, color='#f59e0b', lw=2.5, label=f'Chi-Square Curve (df={df})')

    ax2.fill_between(x, 0, y, where=(x >= chi2_crit), color='#ef4444', alpha=0.5, label=f'Rejection Tail (a = 0.05)')
    ax2.fill_between(x, 0, y, where=(x < chi2_crit), color='#10b981', alpha=0.15, label='Acceptance Zone (Fair Mix)')

    ax2.axvline(chi2_crit, color='#f87171', ls='--', lw=1.5, label=f'Critical X^2 = {chi2_crit:.2f}')
    ax2.axvline(chi2_stat, color='#38bdf8', lw=3, label=f'Audited X^2 = {chi2_stat:.2f}')

    ax2.set_title(f'Chi-Square Distribution (df={df})', color='#f8fafc', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Chi-Square Value', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Probability Density', color='#94a3b8', fontsize=11)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(color='#334155', alpha=0.3, ls=':')
    ax2.legend(facecolor='#1e293b', edgecolor='#f59e0b', labelcolor='#f8fafc', loc='upper right')

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '03_chi_square_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "Chi-Square Goodness-of-Fit",
        "example": "Candy Bag Color Distribution",
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "decision": verdict
    }


if __name__ == '__main__':
    run_candy_color_chi_square_test()
