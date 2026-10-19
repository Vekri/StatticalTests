"""
Vercel / FastAPI entrypoint — Statistical Tests Lab (web)
Mirrors the Streamlit app: load sample CSV or upload, run all 6 tests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
# Prefer samples bundled next to this file (Vercel serverless), then Streamlit folder
_API_SAMPLES = Path(__file__).resolve().parent / "sample_data"
_STREAMLIT_SAMPLES = ROOT / "streamlit_app" / "sample_data"
SAMPLE_DIR = _API_SAMPLES if _API_SAMPLES.exists() else _STREAMLIT_SAMPLES
PUBLIC_DIR = ROOT / "public"

SAMPLES = {
    "z_test": "01_z_test_battery.csv",
    "t_test": "02_t_test_smoothie.csv",
    "chi_square": "03_chi_square_candy.csv",
    "anova": "04_anova_study_methods.csv",
    "correlation": "05_correlation_study_hours.csv",
    "ab_test": "06_ab_test_button.csv",
}

app = FastAPI(title="Statistical Tests Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _df_from_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def _reject(cond: bool) -> str:
    return "REJECT H0" if cond else "FAIL TO REJECT H0"


def _sample_payload(sample_id: str) -> dict[str, Any]:
    if sample_id not in SAMPLES:
        raise HTTPException(404, f"Unknown sample: {sample_id}")
    path = SAMPLE_DIR / SAMPLES[sample_id]
    if not path.exists():
        raise HTTPException(
            404,
            f"Sample file missing on server: {path} (cwd files may be incomplete)",
        )
    df = pd.read_csv(path)
    return {
        "id": sample_id,
        "file": SAMPLES[sample_id],
        "columns": list(df.columns),
        "preview": df.head(15).to_dict(orient="records"),
        "csv": path.read_text(encoding="utf-8"),
        "n_rows": len(df),
    }


@app.get("/")
def home():
    # Prefer bundled HTML next to this file, then public/
    for candidate in (Path(__file__).parent / "index.html", PUBLIC_DIR / "index.html"):
        if candidate.exists():
            return HTMLResponse(candidate.read_text(encoding="utf-8"))
    return JSONResponse({"message": "UI missing. API is up — try /health"})


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "samples": list(SAMPLES.keys()),
        "sample_dir": str(SAMPLE_DIR),
        "sample_dir_exists": SAMPLE_DIR.exists(),
    }


@app.get("/samples")
@app.get("/api/samples")
def list_samples():
    out = []
    for key, fname in SAMPLES.items():
        path = SAMPLE_DIR / fname
        out.append({
            "id": key,
            "file": fname,
            "exists": path.exists(),
            "rows": int(pd.read_csv(path).shape[0]) if path.exists() else 0,
        })
    return {"samples": out}


@app.get("/sample/{sample_id}")
@app.get("/api/sample/{sample_id}")
def get_sample(sample_id: str):
    return _sample_payload(sample_id)


@app.post("/run")
@app.post("/api/run")
async def run_test(
    test: str = Form(...),
    alpha: float = Form(0.05),
    params: str = Form("{}"),
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    test: z_test | t_test | chi_square | anova | correlation | ab_test
    params: JSON string with column names and test-specific knobs
    """
    try:
        cfg = json.loads(params or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid params JSON")

    if file is not None and file.filename:
        raw = (await file.read()).decode("utf-8", errors="replace")
        df = _df_from_csv_text(raw)
        source = file.filename
    elif sample_id:
        if sample_id not in SAMPLES:
            raise HTTPException(400, "Unknown sample_id")
        path = SAMPLE_DIR / SAMPLES[sample_id]
        df = pd.read_csv(path)
        source = SAMPLES[sample_id]
    else:
        raise HTTPException(400, "Provide sample_id or upload a CSV file")

    try:
        if test == "z_test":
            result = _z_test(df, cfg, alpha)
        elif test == "t_test":
            result = _t_test(df, cfg, alpha)
        elif test == "chi_square":
            result = _chi_square(df, cfg, alpha)
        elif test == "anova":
            result = _anova(df, cfg, alpha)
        elif test == "correlation":
            result = _correlation(df, cfg, alpha)
        elif test == "ab_test":
            result = _ab_test(df, cfg, alpha)
        else:
            raise HTTPException(400, f"Unknown test: {test}")
    except KeyError as e:
        raise HTTPException(400, f"Missing column: {e}")
    except Exception as e:
        raise HTTPException(400, str(e))

    result["source"] = source
    result["n_rows"] = int(len(df))
    result["columns"] = list(df.columns)
    result["preview"] = df.head(10).replace({np.nan: None}).to_dict(orient="records")
    return result


def _z_test(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    col = cfg.get("value_col") or df.select_dtypes("number").columns[0]
    mu0 = float(cfg.get("mu0", 24.0))
    sigma = float(cfg.get("sigma", 2.0))
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(x)
    xbar = float(x.mean())
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    reject = abs(z) > z_crit
    return {
        "test": "One-Sample Z-Test",
        "metrics": {
            "n": n, "mean": round(xbar, 4), "mu0": mu0, "sigma": sigma,
            "SE": round(se, 4), "Z": round(z, 4), "critical_Z": round(z_crit, 4),
            "p_value": p,
        },
        "decision": _reject(reject),
        "detail": f"|Z|={abs(z):.3f} vs critical {z_crit:.3f} at alpha={alpha}",
    }


def _t_test(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    col = cfg.get("value_col") or df.select_dtypes("number").columns[0]
    mu0 = float(cfg.get("mu0", 70.0))
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(x)
    t_stat, p = stats.ttest_1samp(x, popmean=mu0)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    reject = abs(t_stat) > t_crit
    return {
        "test": "One-Sample t-Test",
        "metrics": {
            "n": n, "df": n - 1, "mean": round(float(x.mean()), 4),
            "s": round(float(x.std(ddof=1)), 4), "mu0": mu0,
            "t": round(float(t_stat), 4), "critical_t": round(t_crit, 4),
            "p_value": float(p),
        },
        "decision": _reject(reject),
        "detail": f"|t|={abs(t_stat):.3f} vs critical {t_crit:.3f}",
    }


def _chi_square(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    cat_col = cfg.get("cat_col") or df.columns[0]
    count_col = cfg.get("count_col")
    if count_col and count_col in df.columns:
        labels = df[cat_col].astype(str).tolist()
        observed = pd.to_numeric(df[count_col], errors="coerce").dropna().values.astype(float)
    else:
        vc = df[cat_col].value_counts().sort_index()
        labels = vc.index.astype(str).tolist()
        observed = vc.values.astype(float)
    k = len(observed)
    expected = np.full(k, observed.sum() / k)
    chi2, p = stats.chisquare(observed, expected)
    df_chi = k - 1
    chi_crit = float(stats.chi2.ppf(1 - alpha, df_chi))
    reject = chi2 > chi_crit
    return {
        "test": "Chi-Square Goodness-of-Fit",
        "metrics": {
            "k": k, "chi2": round(float(chi2), 4), "critical": round(chi_crit, 4),
            "df": df_chi, "p_value": float(p),
        },
        "table": [
            {"category": lab, "observed": float(o), "expected": float(e)}
            for lab, o, e in zip(labels, observed, expected)
        ],
        "decision": _reject(reject),
        "detail": f"chi2={chi2:.3f} vs critical {chi_crit:.3f}",
    }


def _anova(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    group_col = cfg.get("group_col") or df.columns[0]
    value_col = cfg.get("value_col") or df.select_dtypes("number").columns[0]
    groups = [g[value_col].dropna().astype(float).values for _, g in df.groupby(group_col)]
    labels = [str(n) for n, _ in df.groupby(group_col)]
    f_stat, p = stats.f_oneway(*groups)
    k, n = len(groups), sum(len(g) for g in groups)
    df1, df2 = k - 1, n - k
    f_crit = float(stats.f.ppf(1 - alpha, df1, df2))
    reject = f_stat > f_crit
    summary = (
        df.groupby(group_col)[value_col]
        .agg(["count", "mean", "std"])
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "test": "One-Way ANOVA",
        "metrics": {
            "groups": k, "N": n, "F": round(float(f_stat), 4),
            "critical_F": round(f_crit, 4), "df1": df1, "df2": df2,
            "p_value": float(p),
        },
        "summary": summary,
        "group_labels": labels,
        "decision": _reject(reject),
        "detail": f"F={f_stat:.3f} vs critical {f_crit:.3f}",
    }


def _correlation(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    nums = df.select_dtypes("number").columns.tolist()
    x_col = cfg.get("x_col") or nums[0]
    y_col = cfg.get("y_col") or (nums[1] if len(nums) > 1 else nums[0])
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    r, p = stats.pearsonr(x, y)
    slope, intercept, _, _, se = stats.linregress(x, y)
    reject = p < alpha
    return {
        "test": "Pearson Correlation & Regression",
        "metrics": {
            "n": int(len(x)), "r": round(float(r), 4), "R2_pct": round(float(r ** 2) * 100, 2),
            "p_value": float(p), "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4), "se": round(float(se), 4),
        },
        "formula": f"{y_col} ≈ {intercept:.2f} + {slope:.2f}×{x_col}",
        "decision": _reject(reject),
        "detail": f"H0: rho=0. p={p:.4g}",
    }


def _ab_test(df: pd.DataFrame, cfg: dict, alpha: float) -> dict[str, Any]:
    variant_col = cfg.get("variant_col") or df.columns[0]
    converted_col = cfg.get("converted_col")
    if converted_col and converted_col in df.columns:
        summary = df.groupby(variant_col)[converted_col].agg(["sum", "count"])
        summary.columns = ["successes", "trials"]
    else:
        # assume already aggregated with successes/trials
        success_col = cfg.get("successes_col", "successes")
        trials_col = cfg.get("trials_col", "trials")
        summary = df.set_index(variant_col)[[success_col, trials_col]]
        summary.columns = ["successes", "trials"]

    if len(summary) != 2:
        raise ValueError("A/B test needs exactly 2 variants")

    labels = list(summary.index.astype(str))
    s_a, n_a = int(summary.iloc[0]["successes"]), int(summary.iloc[0]["trials"])
    s_b, n_b = int(summary.iloc[1]["successes"]), int(summary.iloc[1]["trials"])
    p_a, p_b = s_a / n_a, s_b / n_b
    lift = p_b - p_a
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    reject = abs(z) > z_crit
    return {
        "test": "A/B Test (Two-Proportion Z)",
        "metrics": {
            f"{labels[0]}_rate_pct": round(p_a * 100, 3),
            f"{labels[1]}_rate_pct": round(p_b * 100, 3),
            "lift_pp": round(lift * 100, 3),
            "Z": round(float(z), 4),
            "critical_Z": round(z_crit, 4),
            "p_value": p_val,
            "n_a": n_a, "n_b": n_b,
        },
        "variants": labels,
        "decision": _reject(reject),
        "detail": f"|Z|={abs(z):.3f} vs critical {z_crit:.3f}",
    }


# Local static assets (optional)
@app.get("/assets/{name}")
def assets(name: str):
    path = PUBLIC_DIR / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)
