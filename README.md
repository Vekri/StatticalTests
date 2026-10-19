# Statistical Tests — Easy Everyday Video Series

Educational videos + Python code for **6 core statistical tests**, explained with simple real-life examples.

**Repo:** [https://github.com/Vekri/StatticalTests](https://github.com/Vekri/StatticalTests)

---

## Quick start

```bash
pip install -r requirements.txt
python python/01_z_test_easy_example.py
# ... or run any script in /python
```

Videos are in the [`videos/`](videos/) folder. Open any `.mp4` to watch (voice narration included).

---

## All 6 videos (one by one)

### 1. One-Sample Z-Test — Smartphone Battery Claim

| | |
|---|---|
| **Video** | [`videos/01_Z_Test_Easy_Example.mp4`](videos/01_Z_Test_Easy_Example.mp4) |
| **Python** | [`python/01_z_test_easy_example.py`](python/01_z_test_easy_example.py) |
| **Plot** | [`plots/01_z_test_easy_distribution.png`](plots/01_z_test_easy_distribution.png) |

**Description:**  
OmniPhone advertises **24.0 hours** of battery life. A lab tests **n = 64** phones and finds an average of **22.8 hours**. Population σ is known (**2.0 hours**). A one-sample Z-test checks whether the ad claim is wrong.

**Key result:** Z = **−4.80**, p ≈ **1.6×10⁻⁶** → **Reject H₀**. The 24-hour claim is statistically false.

---

### 2. Student's t-Test — Berry Brain Smoothie Quiz

| | |
|---|---|
| **Video** | [`videos/02_T_Test_Easy_Example.mp4`](videos/02_T_Test_Easy_Example.mp4) |
| **Python** | [`python/02_t_test_easy_example.py`](python/02_t_test_easy_example.py) |
| **Plot** | [`plots/02_t_test_easy_distribution.png`](plots/02_t_test_easy_distribution.png) |

**Description:**  
Usual quiz mean is **70**. Only **n = 12** students try a Berry Brain Smoothie and score **76.5** on average (sample s ≈ 5.4). Population σ is **unknown**, so we use Student's t-test (df = 11).

**Key result:** t = **+4.19**, p = **0.0015** → **Reject H₀**. The smoothie boost is real.

---

### 3. Chi-Square Goodness-of-Fit — Candy Bag Colors

| | |
|---|---|
| **Video** | [`videos/03_Chi_Square_Easy_Example.mp4`](videos/03_Chi_Square_Easy_Example.mp4) |
| **Python** | [`python/03_chi_square_easy_example.py`](python/03_chi_square_easy_example.py) |
| **Plot** | [`plots/03_chi_square_easy_distribution.png`](plots/03_chi_square_easy_distribution.png) |

**Description:**  
Candy maker claims equal **20%** of 5 colors. You open **N = 100** candies: Red 15, Blue 18, Green 28, Yellow 14, Orange 25. Chi-square GOF compares observed vs expected (E = 20).

**Key result:** χ² = **7.70** < critical **9.49**, p = **0.103** → **Fail to reject H₀**. Factory mix is fair (uneven bowls = luck).

---

### 4. One-Way ANOVA — 3 Study Methods

| | |
|---|---|
| **Video** | [`videos/04_ANOVA_F_Test_Easy_Example.mp4`](videos/04_ANOVA_F_Test_Easy_Example.mp4) |
| **Python** | [`python/04_anova_f_test_easy_example.py`](python/04_anova_f_test_easy_example.py) |
| **Plot** | [`plots/04_anova_f_easy_distribution.png`](plots/04_anova_f_easy_distribution.png) |

**Description:**  
**60** students split into Textbook, Flashcards, and Interactive Videos (**n = 20** each). ANOVA tests whether study method changes exam scores; Tukey HSD finds which pairs differ.

**Key result:** F ≈ **17.4**, p ≈ **1.3×10⁻⁶** → **Reject H₀**. Flashcards win; Videos also beat Textbook.

---

### 5. Pearson Correlation & Regression — Study Hours vs Grade

| | |
|---|---|
| **Video** | [`videos/05_Correlation_Easy_Example.mp4`](videos/05_Correlation_Easy_Example.mp4) |
| **Python** | [`python/05_correlation_easy_example.py`](python/05_correlation_easy_example.py) |
| **Plot** | [`plots/05_correlation_easy_distribution.png`](plots/05_correlation_easy_distribution.png) |

**Description:**  
Advisor surveys **n = 35** students: weekly study hours (X) vs exam score (Y). Pearson *r* tests association; OLS regression predicts scores.

**Key result:** r ≈ **+0.97**, R² ≈ **94.5%**, p ≈ **0** → **Reject H₀**. Model: **Score ≈ 44 + 2.63 × Hours**.

---

### 6. A/B Testing — Green vs Orange “Buy Now” Button

| | |
|---|---|
| **Video** | [`videos/06_AB_Testing_Easy_Example.mp4`](videos/06_AB_Testing_Easy_Example.mp4) |
| **Python** | [`python/06_ab_testing_easy_example.py`](python/06_ab_testing_easy_example.py) |
| **Plot** | [`plots/06_ab_testing_easy_distribution.png`](plots/06_ab_testing_easy_distribution.png) |

**Description:**  
Randomized A/B test of CTA button color:  
- **A (Green):** 160 / 2,000 = **8.0%** conversion  
- **B (Orange):** 210 / 2,000 = **10.5%** conversion  

Two-proportion Z-test decides if the **+2.5 pp** lift is real.

**Key result:** Z = **+2.73**, p = **0.0064** → **Reject H₀**. Ship Orange to 100% of traffic.

---

## Summary table

| # | Test | Everyday example | Decision |
|---|------|------------------|----------|
| 1 | Z-Test | Phone battery 24h claim | Reject H₀ |
| 2 | t-Test | Berry smoothie quiz (n=12) | Reject H₀ |
| 3 | Chi-Square | Candy color mix | Fail to reject H₀ |
| 4 | ANOVA | 3 study methods | Reject H₀ |
| 5 | Correlation | Study hours vs grade | Reject H₀ |
| 6 | A/B Test | Green vs Orange button | Reject H₀ |

---

## Folder layout

```
StatticalTests/
├── README.md
├── requirements.txt
├── videos/          # 6 narrated MP4 lessons
├── python/          # Matching scipy/statsmodels scripts
└── plots/           # Distribution / chart PNGs
```

## License

Educational use. Feel free to watch, fork, and learn.
