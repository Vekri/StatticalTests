"""
=============================================================================
TEST 4: ONE-WAY ANOVA F-TEST (EASY EVERYDAY EXAMPLE)
Topic: 3 Study Methods (Textbook Reading vs Flashcards vs Interactive Videos)
=============================================================================
Scenario:
    A high school biology teacher tests 60 students preparing for an exam.
    Students are randomly split into 3 equal study groups of 20:
        1. Group A (Textbook): Reading and highlighting the chapter.
        2. Group B (Flashcards): Digital active recall flashcards.
        3. Group C (Videos): Watching animated interactive video lessons.
    Exam results (out of 100):
        - Group A (Textbook)  : Mean = 71.5 pts, Std = 5.8 pts
        - Group B (Flashcards): Mean = 82.1 pts, Std = 6.1 pts
        - Group C (Videos)    : Mean = 77.4 pts, Std = 5.9 pts

Question:
    Does the study technique significantly impact student exam performance,
    or do all 3 methods yield roughly the same outcome? If they differ, which
    method works best?
=============================================================================
"""

import sys
import os
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_study_methods_anova_test():
    print("=" * 75)
    print("TEST 4: ONE-WAY ANOVA F-TEST — 3 STUDY METHODS ON EXAM PERFORMANCE")
    print("=" * 75)

    np.random.seed(42)
    # Generate 20 scores per group matching empirical distributions
    n_per_group = 20
    textbook = np.clip(np.random.normal(loc=71.5, scale=5.8, size=n_per_group), 50, 100)
    flashcards = np.clip(np.random.normal(loc=82.1, scale=6.1, size=n_per_group), 50, 100)
    videos = np.clip(np.random.normal(loc=77.4, scale=5.9, size=n_per_group), 50, 100)

    total_n = len(textbook) + len(flashcards) + len(videos)
    k = 3
    df_between = k - 1
    df_within = total_n - k
    alpha = 0.05

    print(f"Total Students (N)      : {total_n} (k = {k} groups, n = {n_per_group} each)")
    print(f"Degrees of Freedom      : df_between = {df_between}, df_within = {df_within}")
    print("-" * 75)
    print(f"Group 1 (Textbook)      : Mean = {np.mean(textbook):.2f} pts | Std = {np.std(textbook, ddof=1):.2f} pts")
    print(f"Group 2 (Flashcards)    : Mean = {np.mean(flashcards):.2f} pts | Std = {np.std(flashcards, ddof=1):.2f} pts")
    print(f"Group 3 (Interactive)   : Mean = {np.mean(videos):.2f} pts | Std = {np.std(videos, ddof=1):.2f} pts")
    print("-" * 75)

    # 1. One-Way ANOVA Test
    f_stat, p_value = stats.f_oneway(textbook, flashcards, videos)
    f_crit = stats.f.ppf(1 - alpha, df_between, df_within)

    print(f"Calculated F-Statistic  : {f_stat:.3f}")
    print(f"Critical F Threshold    : {f_crit:.3f} (alpha = {alpha})")
    print(f"p-value                 : {p_value:.6f} ({p_value:.2e})")
    print("-" * 75)

    # 2. Decision & Post-Hoc Tukey HSD
    if f_stat > f_crit and p_value < alpha:
        verdict = "REJECT NULL HYPOTHESIS (H0)"
        takeaway = (
            "STUDY TECHNIQUE MATTERS SIGNIFICANTLY!\n"
            f"The F-statistic ({f_stat:.2f}) easily exceeds the critical value ({f_crit:.2f}).\n"
            "At least one study method produces a statistically superior exam score."
        )
    else:
        verdict = "FAIL TO REJECT NULL HYPOTHESIS (H0)"
        takeaway = "No significant difference among the 3 study techniques."

    print(f"DECISION: {verdict}")
    print(takeaway)
    print("-" * 75)

    # Post-hoc pairwise analysis
    all_scores = np.concatenate([textbook, flashcards, videos])
    all_labels = ['Textbook'] * n_per_group + ['Flashcards'] * n_per_group + ['Videos'] * n_per_group
    tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_labels, alpha=alpha)

    print("POST-HOC PAIRWISE TUKEY HSD AUDIT:")
    print(tukey)
    print("=" * 75)

    # 3. Generate Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='#090d16')

    # Subplot 1: Boxplot of the 3 study methods
    ax1.set_facecolor('#0f172a')
    bp = ax1.boxplot([textbook, flashcards, videos], patch_artist=True,
                     tick_labels=['Textbook', 'Flashcards', 'Videos'])
    colors = ['#64748b', '#10b981', '#38bdf8']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#ffffff')
    for median in bp['medians']:
        median.set(color='#f59e0b', linewidth=2.5)

    ax1.set_title('Exam Scores by Study Technique', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exam Score (out of 100)', color='#94a3b8', fontsize=11)
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(color='#334155', alpha=0.3, ls=':', axis='y')

    # Subplot 2: ANOVA F-distribution curve
    ax2.set_facecolor('#0f172a')
    x = np.linspace(0.01, max(20, f_stat + 4), 1000)
    y = stats.f.pdf(x, df_between, df_within)
    ax2.plot(x, y, color='#10b981', lw=2.5, label=f'F-Distribution (df1={df_between}, df2={df_within})')

    ax2.fill_between(x, 0, y, where=(x >= f_crit), color='#ef4444', alpha=0.5, label='Rejection Region (a=0.05)')
    ax2.fill_between(x, 0, y, where=(x < f_crit), color='#10b981', alpha=0.15, label='Acceptance Region')

    ax2.axvline(f_crit, color='#f87171', ls='--', lw=1.5, label=f'Critical F = {f_crit:.2f}')
    ax2.axvline(f_stat, color='#f59e0b', lw=3, label=f'Audited F = {f_stat:.2f}')

    ax2.set_title(f'F-Distribution (df1={df_between}, df2={df_within})', color='#f8fafc', fontsize=12, fontweight='bold')
    ax2.set_xlabel('F-Statistic Value', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Probability Density', color='#94a3b8', fontsize=11)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(color='#334155', alpha=0.3, ls=':')
    ax2.legend(facecolor='#1e293b', edgecolor='#10b981', labelcolor='#f8fafc', loc='upper right')

    out_png = os.path.join(os.path.dirname(__file__), '..', 'plots', '04_anova_f_easy_distribution.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved distribution visual: {out_png}\n")

    return {
        "test": "One-Way ANOVA F-Test",
        "example": "3 Study Methods Comparison",
        "f_stat": f_stat,
        "p_value": p_value,
        "decision": verdict
    }


if __name__ == '__main__':
    run_study_methods_anova_test()
