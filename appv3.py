"""
title: ASCVIT V3 [AUTOMATIC STATISTICAL CALCULATION, VISUALIZATION AND INTERPRETATION TOOL]
author: stefanpietrusky
author_url: https://downchurch.studio/
version: 0.1
"""

import base64
import html
import io
import re
import subprocess
import warnings

from itertools import combinations 

from flask import Flask, request, jsonify
from markupsafe import escape

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.integrate._quadpack_py import IntegrationWarning  

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests

from sklearn.calibration import calibration_curve
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

warnings.filterwarnings('ignore', category=IntegrationWarning)

app = Flask(__name__)
df = None

my_template = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="Arial, sans-serif",
            color="#262626"
        )
    )
)

pio.templates["my_arial"] = my_template
pio.templates.default = "my_arial"

def query_llm_via_cli(input_text, model: str = "llama3.1p"):
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='ignore'
        )
        out, err = proc.communicate(input=input_text + "\n", timeout=40)
        if proc.returncode != 0:
            return f"LLM-Error: {err.strip()}"
        return re.sub(r'\x1b\[.*?m', '', out).strip()
    except Exception as e:
        return f"LLM-Exception: {e}"

def get_llm_interpretation(prompt: str, model: str) -> str:
    try:
        txt = query_llm_via_cli(prompt, model)
        return txt if txt else "(Keine Interpretation erhalten.)"
    except Exception as e:
        return f"(LLM‑Fehler: {e})"

def sanitize_llm_output(text: str) -> str:
    text = re.sub(r'^###\s*(.+)$', r'**\1**', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',   r'<em>\1</em>',     text)

    lines, out, in_list = text.splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "- ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:].strip()}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line)
    if in_list:
        out.append("</ul>")
    text = "\n".join(out)
    text = re.sub(r'(<ul>.*?</ul>)', lambda m: m.group(1).replace('\n', ''), text, flags=re.DOTALL)
    text = text.replace("\n", "<br>")

    placeholders = {
        '<strong>'  : '___TAG_STRONG_OPEN___',
        '</strong>' : '___TAG_STRONG_CLOSE___',
        '<em>'      : '___TAG_EM_OPEN___',
        '</em>'     : '___TAG_EM_CLOSE___',
        '<ul>'      : '___TAG_UL_OPEN___',
        '</ul>'     : '___TAG_UL_CLOSE___',
        '<li>'      : '___TAG_LI_OPEN___',
        '</li>'     : '___TAG_LI_CLOSE___',
        '<br>'      : '___TAG_BR___',
    }
    for tag, ph in placeholders.items():
        text = text.replace(tag, ph)

    text = html.escape(text, quote=False)

    for tag, ph in placeholders.items():
        text = text.replace(ph, tag)

    return text

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_var = ((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / dof
    return (x.mean() - y.mean()) / np.sqrt(pooled_var)

def detect_outliers(X, method='lof', **kwargs):
    if method == 'lof':
        lof = LocalOutlierFactor(**kwargs)
        labels = lof.fit_predict(X)
        mask = labels == 1
        return mask
    return np.ones(len(X), dtype=bool)

def compute_cramers_v(chi2_stat, table):
    n = table.to_numpy().sum()
    return np.sqrt(chi2_stat / (n * (min(table.shape) - 1)))

def detect_date_columns(df: pd.DataFrame, min_success: float = 0.8):
    date_cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(c)
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce")
            success = float(parsed.notna().mean()) if len(parsed) else 0.0
            if success >= min_success and parsed.nunique(dropna=True) > 1:
                date_cols.append(c)
    return date_cols

def _missing_rate(s: pd.Series) -> float:
    return float(s.isna().mean()) if len(s) else 1.0

def _nunique(s: pd.Series) -> int:
    return int(s.dropna().nunique())

def _variance(s: pd.Series) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if len(s2) < 2:
        return 0.0
    return float(np.var(s2, ddof=1))

def _balance_score(counts: pd.Series) -> float:
    if counts.empty:
        return 0.0
    p = counts / counts.sum()
    k = len(p)
    uniform = np.ones(k) / k
    return float(1.0 - np.abs(p.values - uniform).sum() / 2.0)

def detect_date_columns(df: pd.DataFrame, min_success: float = 0.8):
    date_cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(c)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce")
            success = float(parsed.notna().mean()) if len(parsed) else 0.0
            if success >= min_success and parsed.nunique(dropna=True) > 1:
                date_cols.append(c)
    return date_cols

def choose_top_numeric(df: pd.DataFrame, max_cols: int = 8):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    scored = []
    for c in num_cols:
        s = df[c]
        mr = _missing_rate(s)
        nu = _nunique(s)
        var = _variance(s)

        unique_ok = 1.0 if nu >= 5 else (0.5 if nu >= 2 else 0.0)
        score = (1.0 - mr) * 0.55 + unique_ok * 0.20 + (np.log1p(var) / 10.0) * 0.25
        scored.append((c, float(score), mr, nu, var))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_cols]

def choose_top_categorical(df: pd.DataFrame, max_cols: int = 8, max_levels: int = 12):
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    scored = []
    for c in cat_cols:
        s = df[c].dropna()
        mr = _missing_rate(df[c])
        nu = _nunique(df[c])
        if nu < 2:
            continue
        if nu > max_levels:
            continue

        counts = s.value_counts()
        bal = _balance_score(counts)

        level_pref = 1.0 if 2 <= nu <= 6 else (0.7 if 7 <= nu <= 10 else 0.4)
        score = (1.0 - mr) * 0.50 + bal * 0.30 + level_pref * 0.20
        scored.append((c, float(score), mr, nu, bal))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_cols]

def choose_best_corr_pair(df: pd.DataFrame, numeric_candidates: list[str]):
    if len(numeric_candidates) < 2:
        return None
    sub = df[numeric_candidates].copy()
    corr = sub.corr(numeric_only=True).abs()
    if corr.shape[0] < 2:
        return None

    np.fill_diagonal(corr.values, np.nan)
    max_idx = np.nanargmax(corr.values) if np.isfinite(corr.values).any() else None
    if max_idx is None:
        return None

    i, j = np.unravel_index(max_idx, corr.shape)
    a = corr.index[i]
    b = corr.columns[j]
    strength = float(corr.iloc[i, j])
    return (a, b, strength)

def choose_best_regression_target(df: pd.DataFrame, numeric_scored: list[tuple]):
    if not numeric_scored:
        return None
    return numeric_scored[0][0]

def choose_best_binary_target(df: pd.DataFrame, numeric_cols: list[str]):
    best = None
    best_score = -1.0
    for c in numeric_cols:
        s = df[c].dropna()
        if s.nunique() != 2:
            continue
        counts = s.value_counts()
        bal = _balance_score(counts)
        mr = _missing_rate(df[c])
        score = (1.0 - mr) * 0.6 + bal * 0.4
        if score > best_score:
            best_score = score
            best = c
    return best

def choose_best_date(df: pd.DataFrame, date_cols: list[str]):
    best = None
    best_score = -1.0
    for c in date_cols:
        s = df[c]
        mr = _missing_rate(s)
        nu = _nunique(s)
        score = (1.0 - mr) * 0.7 + (min(nu, 365) / 365.0) * 0.3
        if score > best_score:
            best_score = score
            best = c
    return best

def build_overview_recommendations(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    date_cols = detect_date_columns(df)
    num_scored = choose_top_numeric(df, max_cols=10)
    cat_scored = choose_top_categorical(df, max_cols=10, max_levels=12)
    top_num = [c for (c, *_rest) in num_scored]
    top_cat = [c for (c, *_rest) in cat_scored]
    corr_pair = choose_best_corr_pair(df, top_num[:8])
    reg_target = choose_best_regression_target(df, num_scored)
    bin_target = choose_best_binary_target(df, top_num[:10])
    best_date = choose_best_date(df, date_cols) if date_cols else None
    procedures = []
    examples = []

    if top_num:
        procedures.append({
            "name": "Deskriptive Statistik",
            "reason": "Numerische Variablen mit geringer Missing-Rate und ausreichender Varianz gefunden.",
            "features": ["Histogram", "Boxplot"]
        })
        examples.append(f"Histogram/Boxplot für „{top_num[0]}“ (Top-Kandidat nach Datenqualität/Varianz).")

    if corr_pair:
        a, b, strength = corr_pair
        procedures.append({
            "name": "Zusammenhänge (Korrelation/Scatter-Matrix)",
            "reason": "Mindestens zwei hochwertige numerische Variablen; stärkstes |corr|-Paar gewählt.",
            "features": ["Pairplot", "Correlation Matrix"]
        })
        examples.append(f"Korrelation/Scatter: „{a}“ vs. „{b}“ (|r|≈{strength:.2f}).")

    if top_cat and top_num:
        procedures.append({
            "name": "Hypothesentests (Gruppenvergleiche)",
            "reason": "Geeignete Gruppenvariable (moderate Level-Anzahl, balanciert) + hochwertige numerische Variable.",
            "features": ["t-Test", "Mann-Whitney U", "ANOVA", "Welch-ANOVA", "Kruskal-Wallis"]
        })
        g = top_cat[0]
        y = top_num[0]
        nlevels = df[g].dropna().nunique()
        if nlevels == 2:
            lv = df[g].dropna().unique().tolist()[:2]
            examples.append(f"t-Test/Mann-Whitney: „{y}“ zwischen „{g}={lv[0]}“ vs. „{g}={lv[1]}“.")
        else:
            examples.append(f"ANOVA/Kruskal-Wallis: „{y}“ nach Gruppen in „{g}“ ({nlevels} Gruppen).")

    if len(top_cat) >= 2:
        procedures.append({
            "name": "Zusammenhang kategorial × kategorial",
            "reason": "Zwei kategoriale Variablen mit moderater Level-Anzahl und akzeptabler Balance gewählt.",
            "features": ["Chi²-Test", "Fisher Exact (2×2)"]
        })
        a, b = top_cat[0], top_cat[1]
        examples.append(f"Chi²-Test: „{a}“ × „{b}“ (Cramér’s V).")

        if df[a].dropna().nunique() == 2 and df[b].dropna().nunique() == 2:
            examples.append(f"Fisher Exact: „{a}“ × „{b}“ (2×2).")

    if reg_target and len(top_num) >= 2:
        procedures.append({
            "name": "Regression",
            "reason": "Zielvariable als Top-Kandidat nach Varianz/Datenqualität; Prädiktoren aus Top-Numerics.",
            "features": ["Lineare Regression", "Multivariate Regression"]
        })
        preds = [c for c in top_num[:5] if c != reg_target]
        if preds:
            examples.append(f"Regression: Y=„{reg_target}“ mit X={', '.join([f'„{p}“' for p in preds[:3]])} (ggf. + weitere).")
        else:
            examples.append(f"Regression: Y=„{reg_target}“ (Prädiktoren bitte wählen).")

    if bin_target and len(top_num) >= 2:
        procedures.append({
            "name": "Logistische Regression",
            "reason": "Binäres Ziel gewählt, bevorzugt balanciert und mit geringer Missing-Rate.",
            "features": ["Logistische Regression"]
        })
        x = next((c for c in top_num if c != bin_target), None)
        if x:
            examples.append(f"Logistische Regression: Y=„{bin_target}“ mit X=„{x}“ (weitere X möglich).")

    if best_date and top_num:
        procedures.append({
            "name": "Zeitreihenanalyse",
            "reason": "Datumsspalte mit hoher Parse-Qualität + numerischer Top-Kandidat.",
            "features": ["Zeitreihenanalyse"]
        })
        examples.append(f"Zeitreihe: Datum=„{best_date}“, Wert=„{top_num[0]}“ (Trend/Verlauf).")

    if len(top_num) >= 2:
        procedures.append({
            "name": "Clustering",
            "reason": "Mehrere hochwertige numerische Variablen → Cluster-Struktur untersuchen.",
            "features": ["KMeans", "Hierarchical", "DBSCAN"]
        })
        examples.append(f"Clustering mit Variablen: {', '.join([f'„{c}“' for c in top_num[:4]])} (standardisieren empfohlen).")

    return {
        "date_columns": date_cols,
        "binary_numeric": [bin_target] if bin_target else [],
        "suitable_procedures": procedures,
        "analysis_examples": examples,

        "recommended_columns": {
            "top_numeric": top_num[:8],
            "top_categorical": top_cat[:8],
            "best_corr_pair": list(corr_pair[:2]) if corr_pair else [],
            "best_regression_target": reg_target or "",
            "best_binary_target": bin_target or "",
            "best_date": best_date or ""
        },

        "scoring": {
            "numeric": [
                {"col": c, "score": sc, "missing_rate": mr, "nunique": nu, "variance": var}
                for (c, sc, mr, nu, var) in num_scored[:8]
            ],
            "categorical": [
                {"col": c, "score": sc, "missing_rate": mr, "nunique": nu, "balance": bal}
                for (c, sc, mr, nu, bal) in cat_scored[:8]
            ]
        }
    }

# ====================================================================================
# HTML / CSS / JS Template
# ====================================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <title>ASCVIT V3</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      color: #262626;
      padding: 1rem; 
    }
    .button { 
      transition-duration: 0.4s; 
      border: 3px solid #262626; 
      border-radius: 4px; 
      color: white; 
      background-color: #262626; 
      padding: 10px 10px; 
      text-align: center;
      text-decoration: none; 
      display: block; 
      font-size: 16px; 
      margin: 0.5rem 0;
      cursor: pointer; 
    }
    .button1 {
      background-color: #262626; 
      color: white; 
    }
    .button.hover, 
    .button1:hover { 
      background-color: white; 
      color: #262626; 
      border: 3px solid #262626; 
    }
    .sidebar { 
      flex: 0 0 20%;
      padding: 0.5rem; 
    }
    .content { 
      padding: 0.5rem; 
      flex: 1;
    }
    .content .section:empty { 
      display: none; 
    }
    .section { 
      border: 3px solid #262626; 
      padding: 0.5rem; 
      margin-bottom: 1rem; 
      border-radius: 5px; 
    }
    .wrapper {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
    }
    select, input[type="file"], button { 
      display: block; 
      margin: 0.5rem 0; 
      padding: 0.3rem; 
    }
    input[type="checkbox"] { 
      display: inline-block; 
      margin-right: 0.5rem; 
      vertical-align: middle; 
    }
    input[type="file"] {
      padding: 0;
      margin: 0.5rem 0;
      font: inherit; 
      color: inherit;
    }
    input[type="file"]::file-selector-button {
      transition-duration: 0.4s;
      border: 3px solid #262626; 
      border-radius: 4px;
      background-color: #262626;
      color: white;
      padding: 10px 10px;
      margin-right: 4px;
      cursor: pointer;
    }
    input[type="file"]::file-selector-button:hover {
      background-color: white;
      color: #262626;
    }
    input[type="text"] {
      width: 100%;
      box-sizing: border-box;
      padding: 5px;
      font-size: 16px;
      border: 3px solid #262626;
      border-radius: 4px;
    }
    input[type="text"]:focus {  
      outline: none;        
      border-color: #00B0F0;      
    }
    img { 
      max-width: 100%; 
    }
    .info { 
      font-size: 0.9rem; 
      color: #333; 
      margin-bottom: 0.5rem; 
    }
    .checkbox-item { 
      display: flex; 
      align-items: center; 
      margin-bottom: 0.25rem; 
    }
    .checkbox-item input {
      margin-right: 0.5rem; 
    }
    .llm-analysis { 
      margin-top: 1rem; 
      font-size: 0.9rem;
      line-height: 1.4; 
      color: #333;
    }
    .llm-analysis em {
      font-style: normal;
    }
    #clVarBoxContainer { 
      display:flex; 
      flex-direction:column; 
      align-items:flex-start; 
    }
    #clVarBoxContainer .checkbox-item { 
      display:flex; 
      align-items:center; 
      margin-bottom:0.25rem; 
    }
    .tLabel { 
      display:block; 
      margin-bottom: 0.4rem; 
    }
    .tLabel input { 
      margin-left:0.6rem; 
      max-width:150px;
    }
    select {
      display: block;   
      margin: 0.5rem 0;
      padding: 8px 10px;
      font-size: 16px;
      border: 3px solid #262626;
      border-radius: 4px;
      background-color: white;
      color: #262626;
      transition-duration: 0.4s;
      cursor: pointer;
    }
    select:hover {
      background-color: #262626;
      color: white;
    }
    .content table {
      width: auto; 
      max-width: 100%; 
      border-collapse: collapse;  
      margin-bottom: 1rem;  
      table-layout: auto;
    }
    .content table th,
    .content table td {
      border: 1.5px solid #262626; 
      line-height: 1;  
    }
    .content table th {
      text-align: right 
    }
    .content table td {
      text-align: right; 
    }
    .llm-analysis {
      border: 3px solid #262626;
      background-color: #ffffff;
      padding: 0.5rem; 
      margin-top: 0.5rem;
      border-radius: 4px;
      line-height: 1.1;
    }
    .llm-analysis p {
      margin: 0.2rem 0;   
    }
    .llm-analysis ul {
      margin: 0.2rem 0;
      padding-left: 1rem;
    }
    .llm-analysis li {
      margin: 0 0 0.4rem 0;
      padding: 0; 
    }
    .llm-analysis li br {
      display: none;
    }
    #relInfo {
      margin-bottom: 1rem; 
    }
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="sidebar">
      <div class="section">
        <h1>ASCVIT V3</h1>
        <h3>MODELLS</h3>
        <select id="modelSelect">
          <option value="" disabled selected>Lade Modelle…</option>
        </select>
        <h3>UPLOAD</h3>
        <input type="file" id="fileInput" accept=".csv,.xlsx">
        <button id="btnUpload" class="button button1">Hochladen</button>
      </div>

      <div id="overviewSection" class="section" style="display:none;">
        <h3>DATENÜBERSICHT</h3>
        <button id="btnOverview" class="button button1">Einsicht</button>
      </div>

      <div id="descSection" class="section" style="display:none;">
        <h3>DESKRIPTIVE STATISTIK</h3>
        <label>Chart-Typ:
          <select id="descChart">
            <option value="histogram">Histogram</option>
            <option value="boxplot">Boxplot</option>
            <option value="pairplot">Pairplot</option>
            <option value="corrmatrix">Correlation Matrix</option>
          </select>
        </label>

        <div id="descInfo" class="info"></div>

        <div id="descVarsSingle">
          <label>Variable:
            <select id="descVar"></select>
          </label>
        </div>

        <div id="descVarsCheckboxes" style="display:none;">
          <p>Variablen (checkbox):</p>
          <div id="checkboxContainer"></div>
        </div>

        <label id="logSection"><input type="checkbox" id="descLog"> Log-Skalierung</label>
        <button id="btnDesc" class="button button1">Erstelle Diagramm</button>
      </div>

      <div id="hypoSection" class="section" style="display:none;">
        <h3>HYPOTHESENTESTS</h3>

        <label>Test-Typ:
          <select id="hypoTestType">
            <option value="ttest">t-Test</option>
            <option value="mannwhitney">Mann-Whitney U</option>
            <option value="anova">ANOVA</option>
            <option value="welch_anova">Welch-ANOVA</option>
            <option value="kruskal">Kruskal-Wallis</option>
            <option value="anova_rm">ANOVA (Messwiederholung)</option>
            <option value="chi2">Chi²-Test</option>
            <option value="fisher">Fisher Exact (2×2)</option>
          </select>
        </label>

        <div id="hypoInfo" class="info"></div>

        <div id="hypoCommon" style="margin-top:0.5rem;">
          <label>Gruppe:
            <select id="hypoGroup"></select>
          </label>
          <label>Wert:
            <select id="hypoValue"></select>
          </label>
        </div>

        <div id="tParams" style="margin-top:0.5rem;">
          <label class="tLabel">Gruppe&nbsp;1: <input id="hypoG1" type="text"></label>
          <label class="tLabel">Gruppe&nbsp;2: <input id="hypoG2" type="text"></label>
          <label style="display:block; margin-top:0.5rem;">
            <input type="checkbox" id="ttestLog">
            Log-Skalierung (y-Achse)
          </label>
        </div>

        <div id="chi2Params" style="display:none; margin-top:0.5rem;">
          <label>Variable 1:
            <select id="chiVar1"></select>
          </label>
          <label>Variable 2:
            <select id="chiVar2"></select>
          </label>
        </div>

        <div id="rmParams" style="display:none; margin-top:0.5rem;">
          <label>Subject-ID:
            <select id="rmSubject"></select>
          </label>
          <label>Within-Faktor:
            <select id="rmWithin"></select>
          </label>
          <label>Wert:
            <select id="rmValue"></select>
          </label>
        </div>

        <button id="btnHypo" class="button button1">Durchführen</button>
      </div>

      <div id="regSection" class="section" style="display:none;">
        <h3>REGRESSION</h3>
        <div id="regInfo" class="info"></div>
        <label>Typ:
          <select id="regType">
            <option value="linear">Lineare Regression</option>
            <option value="logistic">Logistische Regression</option>
            <option value="multivariate">Multivariate Regression</option>
          </select>
        </label>
        <div id="regYSingle">
          <label>Abhängige Variable (Y):
            <select id="regY"></select>
          </label>
        </div>
        <div id="regYCheckboxes" style="display:none;">
          <p>Abhängige Variable(n) (Y):</p>
          <div id="regYBoxContainer"></div>
        </div>
        <div id="regXCheckboxes">
          <p>Unabhängige Variable(n) (X):</p>
          <div id="regXBoxContainer"></div>
        </div>
        <button id="btnReg" class="button button1">Durchführen</button>
      </div>

      <div id="relSection" class="section" style="display:none;">
        <h3>RELIABILITÄT</h3>
        <div id="relInfo" class="info"></div>
        <p>Items (Checkbox):</p>
        <div id="relItemsBox"></div>
        <button id="btnAlpha" class="button button1">Cronbach's Alpha</button>
      </div>

      <div id="tsSection" class="section" style="display:none;">
        <h3>ZEITREIHENANALYSE</h3>
        <div id="tsInfo" class="info"></div>
        <label>Datum:<select id="tsDate"></select></label>
        <label>Wert:<select id="tsValue"></select></label>
        <button id="btnTS" class="button button1">Durchführen</button>
      </div>

      <div id="clSection" class="section" style="display:none;">
        <h3>CLUSTERING</h3>
        <label>Wähle einen Clustermethode:
          <select id="clMethod">
            <option value="KMeans">KMEANS</option>
            <option value="Hierarchical">HIERARCHICAL</option>
            <option value="DBSCAN">DBSCAN</option>
          </select>
        </label>
        <div id="clInfo" class="info"></div>
        <p>Wähle Variablen aus:</p>
        <div id="clVarBoxContainer" class="checkbox-item"></div>
        <div id="kParam" style="display:none;">
          <p>Wähle die Anzahl an Clustern (k): <span id="kVal">3</span></p>
          <input type="range" id="kSlider" min="2" max="10" step="1" value="3">
        </div>
        <div id="dbParam" style="display:none;">
          <p>Wähle den Radius (eps): <span id="epsVal">0.5</span></p>
          <input type="range" id="epsSlider" min="0.1" max="5" step="0.01" value="0.5">
          <p>Wähle die Mindestanzahl von Punkten pro Cluster (min_samples): <span id="minVal">5</span></p>
          <input type="range" id="minSlider" min="1" max="10" step="1" value="5">
        </div>
        <button id="btnClust" class="button button1">Durchführen</button>
      </div>
    </div>

    <div class="content">
      <div id="outOverview" class="section" style="display:none;"></div>
      <div id="outDesc" class="section"></div>
      <div id="outHypo"  class="section" style="display:none;"></div>
      <div id="outReg" class="section"></div>
      <div id="outRel" class="section" style="display:none;"></div>
      <div id="outTS" class="section"></div>
      <div id="outClust" class="section"></div>
    </div>
  </div>
  <script>
    let allCols = [], numCols = [], catCols = [], binCols = [];

    async function callApi(path, opts) {
      const res = await fetch(path, opts);
      if (!res.ok) {
        let errMsg;
        try {
          const js = await res.json();
          errMsg = js.error || JSON.stringify(js);
        } catch {
          errMsg = await res.text();
        }
        alert("Server-Error:\\n" + errMsg);
        throw new Error(errMsg);
      }
      return res.json();
    }

    document.getElementById('btnOverview').onclick = async () => {
      ['outDesc','outHypo','outReg','outRel','outTS','outClust'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
      });

      const out = document.getElementById('outOverview');
      out.style.display = 'block';
      out.innerHTML = '<p>Die Datenübersicht wird vorbereitet …</p>';
      await new Promise(r => setTimeout(r, 0));

      try {
        const js = await callApi('/overview', { method: 'GET' });

        let html  = js.preview_html;
        html += js.describe_html;

        html += `<p><strong>Number of data points:</strong> ${js.count}</p>`;
        html += `<p><strong>Numerical variables:</strong> ${(js.numerical || []).join(', ')}</p>`;
        html += `<p><strong>Categorical variables:</strong> ${(js.categorical || []).join(', ')}</p>`;

        if (js.date_columns && js.date_columns.length) {
          html += `<p><strong>Date-like columns:</strong> ${js.date_columns.join(', ')}</p>`;
        }
        if (js.binary_numeric && js.binary_numeric.length) {
          html += `<p><strong>Binary numeric (0/1) candidates:</strong> ${js.binary_numeric.join(', ')}</p>`;
        }

        const procs = js.suitable_procedures || [];
        const exs   = js.analysis_examples || [];

        html += `<div class="llm-analysis">`;
        html += `<h3>Geeignete Verfahren (basierend auf Datentypen)</h3>`;

        if (!procs.length) {
          html += `<p>Keine automatischen Empfehlungen verfügbar.</p>`;
        } else {
          html += `<ul>`;
          procs.forEach(p => {
            html += `<li><strong>${p.name}:</strong> ${p.reason}</li>`;
          });
          html += `</ul>`;
        }

        html += `<h3>Beispiele für konkrete Analysen</h3>`;
        if (!exs.length) {
          html += `<p>Keine Beispiele generiert.</p>`;
        } else {
          html += `<ul>`;
          exs.forEach(t => html += `<li>${t}</li>`);
          html += `</ul>`;
        }

        html += `</div>`;

        out.innerHTML = html;
      } catch (e) {
        out.innerHTML = `<p style="color:red;">${e.message || e}</p>`;
      }
    };

    document.getElementById('btnUpload').onclick = async () => {
      try {
        const f = document.getElementById('fileInput').files[0];
        if (!f) throw "Bitte Datei auswählen";
        const fd = new FormData(); fd.append('datafile', f);

        const js = await callApi('/upload', { method:'POST', body: fd });

        allCols = js.columns;
        numCols = js.numerical;
        catCols = js.categorical;
        binCols = js.binary;

        const catCandidates = js.categorical_candidates || catCols || [];

        const groupSel  = document.getElementById('hypoGroup');
        const valueSel  = document.getElementById('hypoValue');
        const chi1Sel   = document.getElementById('chiVar1');
        const chi2Sel   = document.getElementById('chiVar2');
        const rmSubject = document.getElementById('rmSubject');
        const rmWithin  = document.getElementById('rmWithin');
        const rmValue   = document.getElementById('rmValue');

        [groupSel, chi1Sel, chi2Sel].forEach(sel => {
          sel.innerHTML = '';
          catCandidates.forEach(c => sel.append(new Option(c, c)));
        });

        valueSel.innerHTML = '';
        numCols.forEach(c => valueSel.append(new Option(c, c)));

        rmSubject.innerHTML = '';
        allCols.forEach(c => rmSubject.append(new Option(c, c)));

        rmWithin.innerHTML = '';
        catCandidates.forEach(c => rmWithin.append(new Option(c, c)));

        rmValue.innerHTML = '';
        numCols.forEach(c => rmValue.append(new Option(c, c)));

        populateRegressionSelectors('linear');

        ['tsDate','tsValue'].forEach(id => {
          const sel = document.getElementById(id);
          sel.innerHTML = '';
          allCols.forEach(c => sel.append(new Option(c, c)));
        });

        const clBox = document.getElementById('clVarBoxContainer');
        clBox.innerHTML = '';
        numCols.forEach(c => {
          const lbl = document.createElement('label');
          lbl.className = 'checkbox-item';

          const chk = document.createElement('input');
          chk.type  = 'checkbox';
          chk.value = c;

          lbl.append(chk, document.createTextNode(' ' + c));
          clBox.append(lbl);
        });

        const single = document.getElementById('descVar');
        single.innerHTML = '';
        numCols.forEach(c => single.append(new Option(c,c)));

        const cbCont = document.getElementById('checkboxContainer');
        cbCont.innerHTML = '';
        numCols.forEach(c => {
          if (!c) return;

          const lbl = document.createElement('label');
          lbl.className = 'checkbox-item';

          const chk = document.createElement('input');
          chk.type  = 'checkbox';
          chk.value = c;

          const txt = document.createTextNode(c);

          lbl.append(chk, txt);
          cbCont.append(lbl);
        });

        const relBox = document.getElementById('relItemsBox');
        relBox.innerHTML = '';
        numCols.forEach(c => {
          const lbl = document.createElement('label');
          lbl.className = 'checkbox-item';

          const chk = document.createElement('input');
          chk.type = 'checkbox';
          chk.value = c;

          // NEU: Live-Update bei Änderungen
          chk.addEventListener('change', updateRelInfo);

          lbl.append(chk, document.createTextNode(' ' + c));
          relBox.append(lbl);
        });

        document.getElementById('relSection').style.display = 'block';

        updateRelInfo();

        ['overviewSection','descSection','hypoSection',
        'regSection','relSection','tsSection','clSection']
        .forEach(id => {
          const el = document.getElementById(id);
          if (el) el.style.display = 'block';
        });

        document.getElementById('descChart').dispatchEvent(new Event('change'));
        document.getElementById('hypoTestType').dispatchEvent(new Event('change'));
        document.getElementById('regType').dispatchEvent(new Event('change'));
        document.getElementById('clMethod').dispatchEvent(new Event('change'));

      } catch(e) {
        if (typeof e === 'string') alert(e);
        else console.error(e);
      }
    };

    document.getElementById('btnAlpha').onclick = async () => {
      const items = [...document.querySelectorAll('#relItemsBox input:checked')].map(x => x.value);
      if (items.length < 2) return alert('Bitte mindestens 2 Items auswählen');

      const outRel = document.getElementById('outRel');
      outRel.style.display = 'block';
      outRel.innerHTML = '<p>Berechnung läuft …</p>';

      const model = document.getElementById('modelSelect').value;
      const js = await callApi('/cronbach_alpha', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ items, model })
      });

      outRel.innerHTML = (js.html || '');
    };

    const infoMap = {
      histogram: "<strong>Histogram:</strong> Verteilung einer einzelnen numerischen Variable.",
      boxplot: "<strong>Boxplot:</strong> Quartile, Ausreißer und Verteilung einer Variable.",
      pairplot: "<strong>Pairplot:</strong> Scatterplots-Paarweise für mehrere Variablen.",
      corrmatrix: "<strong>Correlation Matrix:</strong> Lineare Zusammenhänge zwischen Variablen."
    };

    const outHypo = document.getElementById('outHypo');

    const hypoInfoMap = {
      ttest: "<strong>t-Test:</strong> Prüft, ob sich die Mittelwerte zweier Gruppen signifikant unterscheiden.",
      mannwhitney: "<strong>Mann-Whitney U:</strong> Nichtparametrischer Vergleich zweier unabhängiger Gruppen (Ränge).",
      anova: "<strong>ANOVA:</strong> Vergleicht die Mittelwerte von drei oder mehr Gruppen auf Unterschiede.",
      welch_anova: "<strong>Welch-ANOVA:</strong> ANOVA-Variante für ungleiche Varianzen (robuster bei Varianzheterogenität).",
      kruskal: "<strong>Kruskal-Wallis:</strong> Nichtparametrischer Gruppenvergleich (Ränge) für 3+ Gruppen.",
      anova_rm: "<strong>ANOVA (Messwiederholung):</strong> Within-Subject-Vergleich für wiederholte Messungen.",
      chi2: "<strong>Chi²-Test:</strong> Untersucht, ob zwei kategoriale Variablen voneinander abhängig sind.",
      fisher: "<strong>Fisher Exact (2×2):</strong> Exakter Test für 2×2-Kontingenztafeln (kleine Stichprobe)."
    };

    const regInfoMap = {
      linear: "<strong>Lineare Regression:</strong> eine numerische abhängige Variable (Y) wird durch eine oder mehrere numerische unabhängige Variable (X) erklärt.",
      logistic: "<strong>Logistische Regression:</strong> die abhängige Variable Y ist binär (0/1) und wird durch eine oder mehrere numerische unabhängige Variable (X) erklärt.",
      multivariate: "<strong>Multivariate Regression:</strong> mehrere numerische abhängige Variablen (Y‑Checkboxen) werden gleichzeitig durch eine oder mehrere numerische unabhängige Variable (X) erklärt."
    };

    const tsInfoText =
      "<strong>Zeitreihenanalyse:</strong> " +
      "zeigt die Entwicklung der gewählten Kennzahl über die Jahre " +
      "und liefert einige zentrale Statistiken (⌀, Min, Max, Streuung).";

    const relInfoText =
      "<strong>Cronbach’s Alpha:</strong> Prüft die interne Konsistenz einer Skala (Reliabilität). " +
      "Wähle mindestens 2 Items aus.";

    function updateRelInfo() {
      const infoEl = document.getElementById('relInfo');
      if (!infoEl) return;

      const selected = [...document.querySelectorAll('#relItemsBox input:checked')]
        .map(x => x.value);

      let status = "";
      if (selected.length === 0) {
        status = "<br><br>Aktuell keine Items ausgewählt.";
      } else if (selected.length === 1) {
        status = `<br><br>Aktuell ausgewählt: ${selected[0]} (mind. 2 nötig).`;
      } else {
        status = `<br><br>Aktuell ausgewählt (${selected.length}): ${selected.join(', ')}`;
      }

      infoEl.innerHTML = relInfoText + status;
    }

    const clInfoMap = {
      KMeans: "<strong>K-Means:</strong> Teilt Daten in K Cluster, indem es die Abstände jeder Beobachtung zu ihrem jeweiligen Cluster-Mittelpunkt minimiert.",
      Hierarchical: "<strong>Hierarchisches Clustering:</strong> Baut sukzessiv Cluster zusammen (oder teilt sie auf) und visualisiert die Verschachtelung in einem Dendrogramm.",
      DBSCAN: "<strong>DBSCAN:</strong> Findet Cluster beliebiger Form anhand lokaler Dichte und kennzeichnet Punkte in zu geringer Dichte als Ausreißer."
    };

    ['kSlider','epsSlider','minSlider'].forEach(id => {
      const slider = document.getElementById(id);
      if (!slider) return;
      const lblId = id === 'kSlider' ? 'kVal' : id === 'epsSlider' ? 'epsVal' : 'minVal';
      const label = document.getElementById(lblId);
      if (label) label.textContent = slider.value;
      slider.addEventListener('input', e => {
        if (label) label.textContent = e.target.value;
      });
    });

    document.getElementById('clMethod').onchange = e => {
      const m = e.target.value;
      document.getElementById('clInfo').innerHTML = clInfoMap[m] || "";
      document.getElementById('kParam').style.display  = (m === 'KMeans' || m === 'Hierarchical') ? 'block':'none';
      document.getElementById('dbParam').style.display = (m === 'DBSCAN') ? 'block':'none';
    };

    document.getElementById('regType').onchange = e => {
      const t = e.target.value;
      document.getElementById('regInfo').innerHTML = regInfoMap[t] || "";

      if (t === 'multivariate') {
        document.getElementById('regYSingle').style.display     = 'none';
        document.getElementById('regYCheckboxes').style.display = 'block';
      } else {
        document.getElementById('regYSingle').style.display     = 'block';
        document.getElementById('regYCheckboxes').style.display = 'none';
      }

      populateRegressionSelectors(t);
    };

    function populateRegressionSelectors(type) {
      const xBox = document.getElementById('regXBoxContainer');
      xBox.innerHTML = '';
      numCols.forEach(c => {
        const lbl = document.createElement('label');
        lbl.className = 'checkbox-item';
        const chk = document.createElement('input');
        chk.type  = 'checkbox';
        chk.value = c;
        lbl.append(chk, document.createTextNode(' ' + c));
        xBox.append(lbl);
      });

      const singleY = document.getElementById('regYSingle');
      const multiY  = document.getElementById('regYCheckboxes');

      const ySel = document.getElementById('regY');
      ySel.innerHTML = '';
      ySel.disabled  = false;

      const yBox = document.getElementById('regYBoxContainer');
      yBox.innerHTML = '';

      if (type === 'multivariate') {
        singleY.style.display = 'none';
        multiY.style.display  = '';
        numCols.forEach(c => {
          const lbl = document.createElement('label');
          lbl.className = 'checkbox-item';
          const chk = document.createElement('input');
          chk.type  = 'checkbox';
          chk.value = c;
          lbl.append(chk, document.createTextNode(' ' + c));
          yBox.append(lbl);
        });

      } else {
        singleY.style.display = '';
        multiY.style.display  = 'none';

        if (type === 'logistic') {
          if (binCols.length) {
            binCols.forEach(c => ySel.append(new Option(c, c)));
          } else {
            const opt = new Option('Keine binäre Variable verfügbar', '');
            opt.disabled = true; opt.selected = true;
            ySel.append(opt);
            ySel.disabled = true;
          }
        }
        else if (type === 'linear') {
          if (numCols.length) {
            numCols.forEach(c => ySel.append(new Option(c, c)));
          } else {
            const opt = new Option('Keine numerische Variable verfügbar', '');
            opt.disabled = true; opt.selected = true;
            ySel.append(opt);
            ySel.disabled = true;
          }
        }
      }
    }

    function injectAndExecuteHTML(target, htmlString) {
      target.innerHTML = htmlString;
      target.querySelectorAll('script').forEach(old => {
        const s = document.createElement('script');
        s.textContent = old.textContent; 
        old.parentNode.replaceChild(s, old);
      });
    }

    document.getElementById('descChart').onchange = e => {
      const t = e.target.value;
      document.getElementById('descInfo').innerHTML = infoMap[t] || "";
      document.getElementById('descVarsSingle').style.display      = (t==='histogram'||t==='boxplot')    ? 'block' : 'none';
      document.getElementById('descVarsCheckboxes').style.display  = (t==='pairplot'||t==='corrmatrix') ? 'block' : 'none';
      document.getElementById('logSection').style.display = (t==='histogram'||t==='boxplot') ? 'block' : 'none';
    };

    // Descriptive Diagramm
    document.getElementById('btnDesc').onclick = async () => {
      document.getElementById('outOverview').style.display = 'none';
      document.getElementById('outReg').style.display = 'none';
      document.getElementById('outTS').style.display = 'none';
      document.getElementById('outClust').style.display = 'none';
      outHypo.style.display = 'none';

      const outDesc = document.getElementById('outDesc');
      outDesc.style.display = 'block';
      outDesc.innerHTML     = '<p>Test wird durchgeführt …</p>';

      const type = document.getElementById('descChart').value;
      const log  = document.getElementById('descLog').checked;
      let vars = [];

      if (type==='histogram'||type==='boxplot') {
        vars = [ document.getElementById('descVar').value ];
      } else {
        document.querySelectorAll('#checkboxContainer input:checked')
          .forEach(chk => vars.push(chk.value));
        if (vars.length < 2) {
          return alert('Bitte mindestens 2 Variablen auswählen');
        }
      }

      const model = document.getElementById('modelSelect').value;
      const js = await callApi('/descriptive', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ chart:type, columns:vars, log:log, model:model })
      });

      let analysisHtml;
      if (js.analysis.startsWith('LLM-Error')) {
        analysisHtml = `<p style="color:red;">${js.analysis}</p>`;
      } else {
        analysisHtml = `
          <div class="llm-analysis">
            <h3>Interpretation:</h3>
            <p>${js.analysis}</p>
          </div>`;
      }

      outDesc.style.display = 'block';
      injectAndExecuteHTML(outDesc, js.plotly_html + analysisHtml);
    };

    // Hypo
    document.getElementById('hypoTestType').onchange = e => {
      const t = e.target.value;
      document.getElementById('hypoInfo').innerHTML = hypoInfoMap[t] || "";
      document.getElementById('rmParams').style.display = (t === 'anova_rm') ? 'block' : 'none';
      document.getElementById('hypoCommon').style.display = (t === 'anova_rm' || t === 'chi2' || t === 'fisher') ? 'none' : 'block';
      document.getElementById('tParams').style.display  = (t === 'ttest' || t === 'mannwhitney') ? 'block' : 'none';
      document.getElementById('chi2Params').style.display = (t === 'chi2' || t === 'fisher') ? 'block' : 'none';
    };

    document.getElementById('btnHypo').addEventListener('click', async () => {
      const model = document.getElementById('modelSelect').value;
      const t = document.getElementById('hypoTestType').value;

      let path, payload;

      if (t === 'ttest') {
        path = '/ttest';
        payload = {
          group_col: document.getElementById('hypoGroup').value,
          value_col: document.getElementById('hypoValue').value,
          group1   : document.getElementById('hypoG1').value.trim(),
          group2   : document.getElementById('hypoG2').value.trim(),
          log      : document.getElementById('ttestLog').checked,
          model    : model
        };
        if (!payload.group1 || !payload.group2) {
          alert('Bitte beide Gruppennamen eingeben');
          return;
        }

      } else if (t === 'mannwhitney') {
        path = '/mannwhitney';
        payload = {
          group_col: document.getElementById('hypoGroup').value,
          value_col: document.getElementById('hypoValue').value,
          group1   : document.getElementById('hypoG1').value.trim(),
          group2   : document.getElementById('hypoG2').value.trim(),
          log      : document.getElementById('ttestLog').checked,
          model    : model
        };
        if (!payload.group1 || !payload.group2) {
          alert('Bitte beide Gruppennamen eingeben');
          return;
        }

      } else if (t === 'anova') {
        path = '/anova';
        payload = {
          group_col: document.getElementById('hypoGroup').value,
          value_col: document.getElementById('hypoValue').value,
          model    : model };

      } else if (t === 'welch_anova') {
        path = '/welch_anova';
        payload = { group_col: document.getElementById('hypoGroup').value,
                    value_col: document.getElementById('hypoValue').value,
                    model: model };

      } else if (t === 'kruskal') {
        path = '/kruskal';
        payload = {
          group_col: document.getElementById('hypoGroup').value,
          value_col: document.getElementById('hypoValue').value,
          model    : model };

      } else if (t === 'anova_rm') {
        path = '/anova_rm';
        payload = {
          subject_col: document.getElementById('rmSubject').value,
          within_col : document.getElementById('rmWithin').value,
          value_col  : document.getElementById('rmValue').value,
          model      : model };

      } else if (t === 'fisher') {
        path = '/fisher';
        payload = {
          var1 : document.getElementById('chiVar1').value,
          var2 : document.getElementById('chiVar2').value,
          model: model };

      } else { 
        path = '/chi2';
        payload = {
          var1 : document.getElementById('chiVar1').value,
          var2 : document.getElementById('chiVar2').value,
          top_n : 10,
          model: model };
      }

      ['outOverview', 'outDesc', 'outReg', 'outTS', 'outClust']
        .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });

      outHypo.style.display = 'block';
      outHypo.innerHTML = '<p>Test wird ausgeführt …</p>';

      try {
        const js = await callApi(path, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(payload)
        });

        injectAndExecuteHTML(outHypo, (js.plotly_html || '') + js.html);
        outHypo.scrollIntoView({ behavior: 'smooth', block: 'start' });

      } catch (err) {
        outHypo.innerHTML = `<p style="color:red;">${err}</p>`;
      }
    });

    // Regression
    document.getElementById('btnReg').onclick = async () => {
      try {

        ['outOverview', 'outDesc', 'outHypo', 'outReg', 'outTS', 'outClust']
          .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });

        const outReg = document.getElementById('outReg');
        outReg.style.display = 'block';
        outReg.innerHTML = '<p>Test wird durchgeführt …</p>';

        const type = document.getElementById('regType').value;

        const getCheckedValues = (selector) =>
          [...document.querySelectorAll(selector)].map(el => el.value);

        let y;
        if (type === 'multivariate') {
          y = getCheckedValues('#regYBoxContainer input:checked');
          if (y.length < 2) throw "Bitte mindestens zwei abhängige Variablen wählen";
        } else {
          y = document.getElementById('regY').value;
          if (!y) {
            alert("Es ist keine geeignete binäre Zielvariable vorhanden.");
            return;
          }
        }

        const x = getCheckedValues('#regXBoxContainer input:checked');
        if (!x.length) throw "Bitte mindestens eine unabhängige Variable ankreuzen";

        const model   = document.getElementById('modelSelect').value;
        const payload = { type, y, x, model };

        const js = await callApi('/regression', {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(payload)
        });
        if (js.error) throw js.error;

        injectAndExecuteHTML(outReg, (js.plotly_html || '') + (js.html || ''));
        outReg.scrollIntoView({ behavior: 'smooth', block: 'start' });

      } catch (e) {
        alert(e);
      }
    };

    // Time Series
    document.getElementById('btnTS').onclick = async () => {
      try {
        ['outOverview','outDesc','outHypo','outReg','outClust']
          .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });

        const outTS = document.getElementById('outTS');
        outTS.style.display = 'block';
        outTS.innerHTML     = '<p>Test wird durchgeführt …</p>';

        const model   = document.getElementById('modelSelect').value;
        const payload = {
          date : document.getElementById('tsDate').value,
          value: document.getElementById('tsValue').value,
          model: model
        };

        const js = await callApi('/timeseries', {
          method : 'POST',
          headers: { 'Content-Type':'application/json' },
          body   : JSON.stringify(payload)
        });

        injectAndExecuteHTML(outTS, js.html);
        outTS.style.display = 'block';
        outTS.scrollIntoView({ behavior:'smooth', block:'start' });

      } catch (e) {
        alert(e.message || e);
      }
    };

    // Clustering
    document.getElementById('btnClust').onclick = async () => {
      try {
        ['outOverview','outDesc','outHypo','outReg','outRel','outTS','outClust']
          .forEach(id => { const el = document.getElementById(id); if (el) el.style.display='none'; });

        const outClust = document.getElementById('outClust');
        outClust.style.display = 'block';
        outClust.innerHTML     = '<p>Test wird durchgeführt …</p>';

        const method = document.getElementById('clMethod').value;
        const vars = [...document.querySelectorAll('#clVarBoxContainer input:checked')]
               .map(chk => chk.value);
        if (!vars.length) throw "Bitte Variablen auswählen";

        const model   = document.getElementById('modelSelect').value;
        const payload = { method, vars, model };

        if (method === 'KMeans' || method === 'Hierarchical')
          payload.k = +document.getElementById('kSlider').value;
        if (method === 'DBSCAN') {
          payload.eps  = +document.getElementById('epsSlider').value;
          payload.min_samples = +document.getElementById('minSlider').value;
        }

        const js = await callApi('/cluster', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (js.error) throw js.error;

        const out = document.getElementById('outClust');
        out.style.display = 'block';

        injectAndExecuteHTML(outClust, (js.plotly_html || '') + js.html);

        out.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch(e){ alert(e.message||e); }
    };

  async function loadModels() {
    try {
      const js = await callApi('/models', { method: 'GET' });
      const sel = document.getElementById('modelSelect');
      sel.innerHTML = '';  // vorher leeren
      js.models.forEach(m => {
        sel.append(new Option(m, m));
      });
      // Default auf das erste Model setzen
      if (js.models.length) sel.value = js.models[0];
    } catch (e) {
      alert("Fehler beim Laden der Modelle: " + e);
    }
  }
  document.addEventListener('DOMContentLoaded', loadModels);

  </script>
</body>
</html>"""
# ====================================================================================
# Flask API Endpoints
# ====================================================================================

@app.route('/')
def home():
    return HTML_PAGE

@app.route('/models', methods=['GET'])
def list_models():
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True
        )
        lines = proc.stdout.splitlines()
        entries = lines[1:]
        model_names = []
        for line in entries:
            parts = line.split()
            if parts:
                model_names.append(parts[0])
        return jsonify(models=model_names)
    except subprocess.CalledProcessError as e:
        return jsonify(error="Konnte Modelle nicht auslesen", details=e.stderr), 500

@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files.get('datafile')
    if file.filename.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    numerical = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    low_card_numeric = [
        c for c in numerical
        if df[c].dropna().nunique() <= 10
    ]
    cat_candidates = sorted(set(categorical + low_card_numeric))

    binary = [c for c in numerical if df[c].dropna().nunique() == 2]

    return jsonify(
        columns=df.columns.tolist(),
        numerical=numerical,
        categorical=categorical,
        categorical_candidates=cat_candidates,
        binary=binary
    )

@app.route('/overview', methods=['GET'])
def overview():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    preview_html  = df.head().to_html(classes="table table-sm", border=0, index=False)
    describe_html = df.describe().to_html(classes="table table-sm", border=0)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    rec = build_overview_recommendations(df)

    return jsonify(
        preview_html    = preview_html,
        describe_html   = describe_html,
        count           = len(df),
        numerical       = num_cols,
        categorical     = cat_cols,
        date_columns        = rec["date_columns"],
        binary_numeric      = rec["binary_numeric"],
        suitable_procedures = rec["suitable_procedures"],
        analysis_examples   = rec["analysis_examples"],
        recommended_columns = rec["recommended_columns"],
        scoring             = rec["scoring"]
    )

@app.route('/descriptive', methods=['POST'])
def descriptive():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    params = request.get_json()
    cols  = params['columns']
    log   = params.get('log', False)
    chart = params['chart']

    sns.set_theme(style="whitegrid")
    context = ""

    hue_col = "Kategorie"

    if chart == 'histogram':
        data = df[cols[0]].dropna()
        mean, median = data.mean(), data.median()
        std, mn, mx = data.std(), data.min(), data.max()
        skew = st.skew(data)
        kurt = st.kurtosis(data, fisher=True)
        ci_low, ci_high = st.t.interval(0.95, len(data)-1,
                                        loc=mean, scale=st.sem(data))

        fig = px.histogram(
            df, x=cols[0],
            nbins=30,
            title=f"Histogramm von {cols[0]}",
            labels={cols[0]: cols[0]}
        )
        if log:
            fig.update_yaxes(type="log")

        fig.add_vline(
            x=mean,
            line_dash="dash",
            annotation_text=f"Mittelwert: {mean:.2f}",
            annotation_position="top left"
        )
        fig.add_vline(
            x=median,
            line_dash="dot",
            annotation_text=f"Median: {median:.2f}",
            annotation_position="top right"
        )

        stats_text = (
            f"M: {mean:.2f}<br>"
            f"Med: {median:.2f}<br>"
            f"SD: {std:.2f}<br>"
            f"Skew: {skew:.2f}<br>"
            f"Kurt: {kurt:.2f}<br>"
            f"CI95%: [{ci_low:.2f}, {ci_high:.2f}]"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.60, y=0.75,
            text=stats_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            opacity=0.8,
            align="left"
        )

        context = (
            f"Analyse der Verteilung von '{cols[0]}':\n"
            f"- Mittelwert: {mean:.2f}\n"
            f"- Median: {median:.2f}\n"
            f"- Std-Abweichung: {std:.2f}\n"
            f"- Schiefe: {skew:.2f}\n"
            f"- Wölbung: {kurt:.2f}\n"
            f"- 95%-CI Mittelwert: [{ci_low:.2f}, {ci_high:.2f}]\n"
            "Beschreibe Form, Streuung und Ausreißer. Keine Meta-Antworten!"
        )

    elif chart == 'boxplot':
        data = df[cols[0]].dropna()
        median = data.median()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        lw = max(data.min(), q1 - 1.5 * iqr)
        uw = min(data.max(), q3 + 1.5 * iqr)
        skew = st.skew(data)
        kurt = st.kurtosis(data, fisher=True)
        outliers = data[(data < lw) | (data > uw)]
        n_out = len(outliers)

        fig = px.box(
            df,
            y=cols[0],
            points="outliers",
            title=f"Boxplot von {cols[0]}",
            labels={cols[0]: cols[0]}
        )
        if log:
            fig.update_yaxes(type="log")

        stats_text = (
            f"Med: {median:.2f}<br>"
            f"Q1: {q1:.2f}, Q3: {q3:.2f}<br>"
            f"IQR: {iqr:.2f}<br>"
            f"Whiskers: [{lw:.2f}, {uw:.2f}]<br>"
            f"Ausreißer: {n_out}<br>"
            f"Skew: {skew:.2f}<br>"
            f"Kurt: {kurt:.2f}"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            text=stats_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            opacity=0.8,
            align="right"
        )

        context = (
            f"Boxplot zu '{cols[0]}':\n"
            f"- Median: {median:.2f}\n"
            f"- Q1: {q1:.2f}, Q3: {q3:.2f}\n"
            f"- IQR: {iqr:.2f}\n"
            f"- Whiskers: [{lw:.2f}, {uw:.2f}]\n"
            f"- Ausreißer: {n_out} Werte\n"
            f"- Schiefe: {skew:.2f}\n"
            f"- Wölbung: {kurt:.2f}\n"
            "Identifiziere Ausreißer und beschreibe die Verteilung. Keine Meta-Antworten!"
        )

    elif chart == 'pairplot':
        data = df[cols + ([hue_col] if hue_col in df.columns else [])].dropna()

        fig = px.scatter_matrix(
            data,
            dimensions=cols,
            color=hue_col if hue_col in data else None,
            title=f"Scatter Matrix: {', '.join(cols)}",
            labels={c: c for c in cols}
        )
        fig.update_traces(diagonal_visible=False)

        corr = df[cols].corr().round(2)
        corr_list = "\n".join(
            f"- {v1} vs. {v2}: {corr.at[v1, v2]:.2f}"
            for v1 in corr.columns for v2 in corr.columns if v1 < v2
        )
        context = (
            f"Paarweise Scatterplots für {', '.join(cols)}"
            + (f" gefärbt nach '{hue_col}'" if hue_col in data else "")
            + ":\n"
            + corr_list
            + "\nBeschreibe die wichtigsten Beziehungen. Keine Meta-Antworten!"
        )

    elif chart == 'corrmatrix':
        cm = df[cols].corr()

        pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
        for i, x in enumerate(cols):
            for j, y in enumerate(cols):
                if i < j:
                    xi = df[x].dropna()
                    yi = df[y].dropna()
                    zi = pd.concat([xi, yi], axis=1).dropna()
                    r, p = st.pearsonr(zi[x], zi[y])
                    pvals.at[x, y] = p
                    pvals.at[y, x] = p

        cis = {}
        for i, x in enumerate(cols):
            for j, y in enumerate(cols):
                if i < j:
                    r = cm.at[x, y]
                    n = df[[x, y]].dropna().shape[0]
                    z = np.arctanh(r)
                    se = 1 / np.sqrt(n - 3)
                    lo, hi = np.tanh((z - 1.96 * se, z + 1.96 * se))
                    cis[(x, y)] = (lo, hi)

        strong = []
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if i < j and abs(cm.at[a, b]) >= 0.5:
                    r = cm.at[a, b]
                    pval = pvals.at[a, b]
                    lo, hi = cis[(a, b)]
                    r2 = r**2
                    strong.append(
                        f"- {a} vs. {b}: r={r:.2f}, p={pval:.2g}, 95% CI [{lo:.2f}, {hi:.2f}], R²={r2:.2f}"
                    )
        corr_list = "\n".join(strong) if strong else "Keine signifikanten Korrelationen (|r|<0.5)."

        context = (
            f"Korrelationsanalyse (n={len(df.dropna(subset=cols))}):\n"
            f"{corr_list}\n"
            "Gib eine statistische Interpretation jeder starken Korrelation. Keine Meta-Antworten!"
        )

        fig = px.imshow(
            cm,
            color_continuous_scale="RdBu_r",
            origin="lower",
            title="Korrelationsmatrix",
            labels=dict(x="Variable", y="Variable", color="r"),
            width=750,
            height=750
        )

        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
        text = [
            [f"{cm.iloc[i, j]:.2f}" if not mask[i, j] else "" for j in range(len(cols))]
            for i in range(len(cols))
        ]

        fig.update_traces(text=text, texttemplate="%{text}", 
                          hovertemplate="r=%{z:.2f}<extra></extra>")

        fig.update_xaxes(
            tickangle=45, 
            tickfont=dict(size=12), 
            automargin=True  
        )
        fig.update_yaxes(
            tickangle=0,      
            tickfont=dict(size=12),
            automargin=True   
        )

    else:
        return jsonify(error="Unbekannter Chart-Typ"), 400

    model = params.get('model', 'llama3.1p')
    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)
    raw_analysis = get_llm_interpretation(context, model)
    analysis     = sanitize_llm_output(raw_analysis)

    return jsonify(plotly_html=plotly_html, analysis=analysis)

@app.route('/ttest', methods=['POST'])
def ttest():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p = request.json
    model = p.get('model', 'llama3.1p')
    grp = p['group_col']
    val = p['value_col']
    g1, g2 = p['group1'], p['group2']
    log_scale = p.get('log', False)

    data1 = df[df[grp] == g1][val].dropna()
    data2 = df[df[grp] == g2][val].dropna()
    total1, n1 = len(df[df[grp] == g1]), len(data1)
    total2, n2 = len(df[df[grp] == g2]), len(data2)
    if n1 == 0 or n2 == 0:
        return jsonify(error="Gruppendaten fehlen"), 400

    w_levene, p_levene = st.levene(data1, data2)
    equal_var = p_levene >= 0.05
    t_stat, p_val   = st.ttest_ind(data1, data2, equal_var=equal_var)
    d               = cohens_d(data1, data2)
    diff = data1.mean() - data2.mean()
    se_diff = np.sqrt(data1.var(ddof=1)/n1 + data2.var(ddof=1)/n2)
    df_dof = n1 + n2 - 2
    ci_low, ci_high = st.t.interval(0.95, df=df_dof, loc=diff, scale=se_diff)
    s1, p_s1 = st.shapiro(data1)
    s2, p_s2 = st.shapiro(data2)

    power = TTestIndPower().solve_power(effect_size=d, nobs1=n1, alpha=0.05, ratio=n2/n1)

    prompt = (
        f"Zweiseitiger t-Test zwischen {g1} (n={n1}, μ={data1.mean():.2f}) "
        f"und {g2} (n={n2}, μ={data2.mean():.2f}): t={t_stat:.3f}, p={p_val:.3e}, "
        f"Cohen's d={d:.2f}, 95%-CI=[{ci_low:.2f},{ci_high:.2f}], "
        f"Levene p={p_levene:.3f}, Shapiro p1={p_s1:.3f}, p2={p_s2:.3f}, Power={power:.2f}."
        f"Keine Meta-Antworten!"
    )
    raw = get_llm_interpretation(prompt, model)
    interpretation = sanitize_llm_output(raw)

    def count_outliers(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    out1, out2 = count_outliers(data1), count_outliers(data2)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Boxplot', 'Violinplot', f'Q-Q {g1}', f'Q-Q {g2}')
    )

    fig.add_trace(
        go.Box(y=data1, name=g1, boxpoints='outliers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=data2, name=g2, boxpoints='outliers'),
        row=1, col=1
    )

    fig.add_trace(
        go.Violin(y=data1, name=g1, points='all', jitter=0.3),
        row=1, col=2
    )
    fig.add_trace(
        go.Violin(y=data2, name=g2, points='all', jitter=0.3),
        row=1, col=2
    )

    osm1, osr1 = st.probplot(data1, dist='norm')[0]
    fig.add_trace(
        go.Scatter(x=osr1, y=osr1, mode='lines', name=f'45° Linie {g1}'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=osm1, y=osr1, mode='markers', name=g1),
        row=2, col=1
    )

    osm2, osr2 = st.probplot(data2, dist='norm')[0]
    fig.add_trace(
        go.Scatter(x=osr2, y=osr2, mode='lines', name=f'45° Linie {g2}'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=osm2, y=osr2, mode='markers', name=g2),
        row=2, col=2
    )

    if log_scale:
        fig.update_yaxes(type="log")

    fig.update_layout(
        height=800, width=800,
        title_text=f"Gruppenvergleich: {g1} vs {g2}",
        showlegend=True
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    mw_html = ""
    if mannwhitney is not None:
        u_stat, p_u = mannwhitney
        mw_html = f"<p><strong>Mann-Whitney U:</strong> U={u_stat:.2f}, p={p_u:.4e}</p>"

    html = (
        f"<h4>Gruppenvergleich</h4>"
        f"<p><strong>{g1}</strong> (gesamt: {total1}, n={n1}) vs. "
        f"<strong>{g2}</strong> (gesamt: {total2}, n={n2})</p>"
        f"<p><strong>t-Statistik:</strong> {t_stat:.4f}, "
        f"<strong>p-Wert:</strong> {p_val:.4e}</p>"
        f"<p><strong>Cohen's d:</strong> {d:.3f}</p>"
        f"<p><strong>95%-CI Differenz:</strong> [{ci_low:.2f}, {ci_high:.2f}]</p>"
        f"<p><strong>Levene p-Wert:</strong> {p_levene:.3f}</p>"
        f"<p><strong>Shapiro p-Werte:</strong> {p_s1:.3f}, {p_s2:.3f}</p>"
        f"<p><strong>Power:</strong> {power:.2f}</p>"
        f"<h4>Deskriptive Kennzahlen</h4>"
        f"<p>{g1}: Median={data1.median():.2f}, IQR={(data1.quantile(0.75)-data1.quantile(0.25)):.2f}, "
        f"σ={data1.std(ddof=1):.2f}, Skew={data1.skew():.2f}, Kurtosis={data1.kurtosis():.2f}</p>"
        f"<p>{g2}: Median={data2.median():.2f}, IQR={(data2.quantile(0.75)-data2.quantile(0.25)):.2f}, "
        f"σ={data2.std(ddof=1):.2f}, Skew={data2.skew():.2f}, Kurtosis={data2.kurtosis():.2f}</p>"
        f"<h4>Outlier</h4><p>{g1}: {out1} Ausreißer; {g2}: {out2} Ausreißer</p>"
        f"<h4>Interpretation vom LLM</h4><p>{interpretation}</p>"
    )

    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/mannwhitney', methods=['POST'])
def mannwhitney():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p = request.json
    model = p.get('model', 'llama3.1p')
    grp = p['group_col']
    val = p['value_col']
    g1, g2 = p['group1'], p['group2']
    log_scale = p.get('log', False)

    data1 = df[df[grp] == g1][val].dropna()
    data2 = df[df[grp] == g2][val].dropna()
    total1, n1 = len(df[df[grp] == g1]), len(data1)
    total2, n2 = len(df[df[grp] == g2]), len(data2)
    if n1 == 0 or n2 == 0:
        return jsonify(error="Gruppendaten fehlen"), 400

    u_stat, p_u = st.mannwhitneyu(data1, data2, alternative="two-sided")

    cliffs_delta = (2 * u_stat) / (n1 * n2) - 1

    mean_u = n1 * n2 / 2
    sd_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_u) / sd_u if sd_u > 0 else np.nan
    r = abs(z) / np.sqrt(n1 + n2) if (n1 + n2) > 0 else np.nan

    w_levene, p_levene = st.levene(data1, data2)
    s1, p_s1 = st.shapiro(data1)
    s2, p_s2 = st.shapiro(data2)

    prompt = (
        f"Mann-Whitney-U-Test zwischen {g1} (n={n1}, Median={data1.median():.2f}) "
        f"und {g2} (n={n2}, Median={data2.median():.2f}): U={u_stat:.3f}, p={p_u:.3e}, "
        f"Cliff's delta={cliffs_delta:.3f}, r≈{r:.3f}. "
        f"Levene p={p_levene:.3f}, Shapiro p1={p_s1:.3f}, p2={p_s2:.3f}. "
        f"Keine Meta-Antworten!"
    )
    raw = get_llm_interpretation(prompt, model)
    interpretation = sanitize_llm_output(raw)

    def count_outliers(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    out1, out2 = count_outliers(data1), count_outliers(data2)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Boxplot', 'Violinplot', f'Q-Q {g1}', f'Q-Q {g2}')
    )

    fig.add_trace(go.Box(y=data1, name=g1, boxpoints='outliers'), row=1, col=1)
    fig.add_trace(go.Box(y=data2, name=g2, boxpoints='outliers'), row=1, col=1)

    fig.add_trace(go.Violin(y=data1, name=g1, points='all', jitter=0.3), row=1, col=2)
    fig.add_trace(go.Violin(y=data2, name=g2, points='all', jitter=0.3), row=1, col=2)

    osm1, osr1 = st.probplot(data1, dist='norm')[0]
    fig.add_trace(go.Scatter(x=osr1, y=osr1, mode='lines', name=f'45° Linie {g1}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=osm1, y=osr1, mode='markers', name=g1), row=2, col=1)

    osm2, osr2 = st.probplot(data2, dist='norm')[0]
    fig.add_trace(go.Scatter(x=osr2, y=osr2, mode='lines', name=f'45° Linie {g2}'), row=2, col=2)
    fig.add_trace(go.Scatter(x=osm2, y=osr2, mode='markers', name=g2), row=2, col=2)

    if log_scale:
        fig.update_yaxes(type="log")

    fig.update_layout(
        height=800, width=800,
        title_text=f"Gruppenvergleich (Mann-Whitney U): {g1} vs {g2}",
        showlegend=True
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    html = (
        f"<h4>Gruppenvergleich (Mann-Whitney U)</h4>"
        f"<p><strong>{g1}</strong> (gesamt: {total1}, n={n1}) vs. "
        f"<strong>{g2}</strong> (gesamt: {total2}, n={n2})</p>"
        f"<p><strong>U:</strong> {u_stat:.4f}, <strong>p-Wert:</strong> {p_u:.4e}</p>"
        f"<p><strong>Cliff's delta:</strong> {cliffs_delta:.3f} &nbsp; "
        f"<strong>r (Approx.):</strong> {r:.3f}</p>"
        f"<p><strong>Levene p-Wert:</strong> {p_levene:.3f}</p>"
        f"<p><strong>Shapiro p-Werte:</strong> {p_s1:.3f}, {p_s2:.3f}</p>"
        f"<h4>Deskriptive Kennzahlen</h4>"
        f"<p>{g1}: Median={data1.median():.2f}, IQR={(data1.quantile(0.75)-data1.quantile(0.25)):.2f}, "
        f"σ={data1.std(ddof=1):.2f}, Skew={data1.skew():.2f}, Kurtosis={data1.kurtosis():.2f}</p>"
        f"<p>{g2}: Median={data2.median():.2f}, IQR={(data2.quantile(0.75)-data2.quantile(0.25)):.2f}, "
        f"σ={data2.std(ddof=1):.2f}, Skew={data2.skew():.2f}, Kurtosis={data2.kurtosis():.2f}</p>"
        f"<h4>Outlier</h4><p>{g1}: {out1} Ausreißer; {g2}: {out2} Ausreißer</p>"
        f"<h4>Interpretation vom LLM</h4><p>{interpretation}</p>"
    )

    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/anova', methods=['POST'])
def anova():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p      = request.json
    model  = p.get('model', 'llama3.1p')
    grp    = p['group_col']
    val    = p['value_col']
    data   = df[[grp, val]].dropna()

    groups      = [g[val].values for _, g in data.groupby(grp)]
    k           = len(groups)
    if k < 2:
        return jsonify(error="Zu wenige Gruppen"), 400

    w_levene, p_levene = st.levene(*groups)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        f_stat, p_val = st.f_oneway(*groups)

    overall_mean = data[val].mean()
    ss_between   = sum(len(g) * (g.mean() - overall_mean)**2 for g in groups)
    ss_within    = sum(((g - g.mean())**2).sum() for g in groups)
    eta2         = ss_between / (ss_between + ss_within)
    power        = FTestAnovaPower().solve_power(
                       effect_size=np.sqrt(eta2/(1-eta2)),
                       k_groups=k, nobs=data.shape[0], alpha=0.05
                   )

    group_means = data.groupby(grp)[val].transform('mean')
    residuals   = data[val] - group_means
    s_res, p_res = st.shapiro(residuals)

    posthoc_html = ""
    if p_val < 0.05:
        mc    = MultiComparison(data[val], data[grp])
        tukey = mc.tukeyhsd(alpha=0.05)
        means = data.groupby(grp)[val].mean().sort_values()
        try:
            cld_df = tukey.get_cld()
            cld     = dict(zip(cld_df.group, cld_df.clds))
        except:
            cld = {g: '' for g in means.index}

        fig_cld = go.Figure()
        fig_cld.add_trace(go.Bar(
            x=means.values,
            y=means.index,
            orientation='h',
            marker_color='lightgrey',
            text=[cld[g] for g in means.index],
            textposition='outside'
        ))
        fig_cld.update_layout(
            title="Gruppenmittelwerte mit CLD",
            xaxis_title="Mittelwert",
            yaxis_title=grp,
            height= max(200, 30*len(means)),
            margin=dict(l=100, r=40, t=60, b=40),
            showlegend=False
        )
        posthoc_html += "<h4>Post-hoc: Gruppengruppierung (CLD)</h4>"
        posthoc_html += fig_cld.to_html(full_html=False, include_plotlyjs=False)

        res        = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        sig        = res[res['reject'] == True]
        if not sig.empty:
            fig_sig = go.Figure()
            y_pos   = list(sig.index)
            fig_sig.add_trace(go.Scatter(
                x=sig['meandiff'],
                y=y_pos,
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=sig['upper'] - sig['meandiff'],
                    arrayminus=sig['meandiff'] - sig['lower']
                ),
                mode='markers'
            ))
            fig_sig.update_layout(
                title="Signifikante Post-hoc-Paare",
                xaxis_title="Mittelwertsdifferenz",
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_pos,
                    ticktext=[f"{g1}-{g2}" for g1, g2 in zip(sig['group1'], sig['group2'])]
                ),
                height= max(200, 30*len(sig)),
                margin=dict(l=150, r=40, t=60, b=40),
                showlegend=False
            )
            posthoc_html += "<h4>Signifikante Paare</h4>"
            posthoc_html += fig_sig.to_html(full_html=False, include_plotlyjs=False)
            posthoc_html += "<div style='max-height:200px; overflow:auto;'>"
            posthoc_html += sig.to_html(index=False, classes='table table-sm', border=0)
            posthoc_html += "</div>"

    means_dict = data.groupby(grp)[val].mean().round(2).to_dict()
    prompt = (
        f"ANOVA mit {k} Gruppen ({grp}): F={f_stat:.3f}, p={p_val:.3e}. "
        f"Levene p={p_levene:.3f} -> {'Varianzhomogenität' if p_levene>=0.05 else 'Varianzheterogenität'}. "
        f"Shapiro-Wilk Residuen p={p_res:.3f} -> {'normalverteilt' if p_res>=0.05 else 'Nicht-Normalität'}. "
        f"Eta²={eta2:.3f}, Power={power:.3f}. Mittelwerte: {means_dict}. "
        f"Fasse in 1 Satz zusammen, ob Voraussetzungen erfüllt sind, der Effekt signifikant ist und wie Post-hocs interpretiert werden sollten."
        f"Keine Meta-Antworten!"
    )
    raw  = get_llm_interpretation(prompt, model)
    interp = sanitize_llm_output(raw)

    fig = make_subplots(rows=1, cols=3, subplot_titles=('Boxplot', 'Violinplot', 'Q-Q Residuen'))

    fig.add_trace(
        go.Box(y=data[val], x=data[grp], name='', boxpoints='outliers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Violin(y=data[val], x=data[grp], name='', points='all', jitter=0.3),
        row=1, col=2
    )
    osm, osr = st.probplot(residuals, dist='norm')[0]
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuen'), row=1, col=3)
    fig.add_trace(go.Scatter(x=osm, y=osm, mode='lines', name='45° Linie'),  row=1, col=3)

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, width=1200, showlegend=False, title_text="ANOVA: Verteilung & Residuen")

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    html = (
        f"<h4>ANOVA-Ergebnisse</h4>"
        f"<p>Gruppen: {k}, F={f_stat:.3f}, p={p_val:.3e}</p>"
        f"<p>Levene p-Wert: {p_levene:.3f} "
        f"({'Varianzhomogenität' if p_levene>=0.05 else 'Varianzheterogenität'})</p>"
        f"<p>Eta²: {eta2:.3f}, Power: {power:.3f}</p>"
        f"<p>Shapiro-Wilk Residuen: W={s_res:.3f}, p={p_res:.3f} "
        f"({'normalverteilt' if p_res>=0.05 else 'abweichend'})</p>"
        + posthoc_html +
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"
    )

    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/welch_anova', methods=['POST'])
def welch_anova():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p     = request.json or {}
    model = p.get('model', 'llama3.1p')
    grp   = p['group_col']
    val   = p['value_col']

    data = df[[grp, val]].dropna()
    if data[grp].nunique() < 2:
        return jsonify(error="Zu wenige Gruppen"), 400

    res = anova_oneway(data[val], groups=data[grp], use_var="unequal")
    F_val = float(res.statistic)
    p_val = float(res.pvalue)

    groups = {g: d[val].dropna().values for g, d in data.groupby(grp)}
    pairs, pvals, diffs = [], [], []
    for g1, g2 in combinations(groups.keys(), 2):
        t_stat, p_pair = st.ttest_ind(groups[g1], groups[g2], equal_var=False)
        pairs.append((g1, g2))
        pvals.append(p_pair)
        diffs.append(np.mean(groups[g1]) - np.mean(groups[g2]))

    rej, p_adj, *_ = multipletests(pvals, alpha=0.05, method="holm")
    posthoc = pd.DataFrame({
        "group1": [a for a, b in pairs],
        "group2": [b for a, b in pairs],
        "mean_diff": diffs,
        "p_raw": pvals,
        "p_holm": p_adj,
        "reject": rej
    }).sort_values("p_holm")

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Boxplot', 'Violinplot'))
    fig.add_trace(go.Box(y=data[val], x=data[grp], boxpoints='outliers'), row=1, col=1)
    fig.add_trace(go.Violin(y=data[val], x=data[grp], points='all', jitter=0.3), row=1, col=2)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, width=900, showlegend=False, title_text="Welch-ANOVA")

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    prompt = (
        f"Welch-ANOVA ({grp} -> {val}): F={F_val:.3f}, p={p_val:.3e}. "
        f"Post-hoc (Welch t + Holm): Top-Ergebnisse: "
        f"{posthoc.head(5).to_dict(orient='records')}. "
        "Erkläre kurz, ob Gruppen sich unterscheiden und welche Paare relevant sind. Keine Meta-Antworten!"
    )
    interp = sanitize_llm_output(get_llm_interpretation(prompt, model))

    html = (
        "<h4>Welch-ANOVA</h4>"
        f"<p>F={F_val:.3f}, p={p_val:.3e}</p>"
        "<h4>Post-hoc (Welch t, Holm-korrigiert)</h4>"
        f"<div style='max-height:220px; overflow:auto;'>{posthoc.to_html(index=False, border=0)}</div>"
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"
    )
    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/anova_rm', methods=['POST'])
def anova_rm():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p     = request.json or {}
    model = p.get('model', 'llama3.1p')
    subj  = p['subject_col']
    within= p['within_col']
    val   = p['value_col']

    data = df[[subj, within, val]].dropna()

    if data[within].nunique() < 2:
        return jsonify(error="Within-Faktor braucht mind. 2 Stufen"), 400
    if data[subj].nunique() < 2:
        return jsonify(error="Zu wenige Subjekte"), 400

    aov = AnovaRM(data, depvar=val, subject=subj, within=[within]).fit()
    tab = aov.anova_table.copy()

    if {'F Value','Num DF','Den DF'}.issubset(tab.columns):
        F  = tab['F Value'].iloc[0]
        df1= tab['Num DF'].iloc[0]
        df2= tab['Den DF'].iloc[0]
        eta_p2 = (F*df1) / (F*df1 + df2) if (F*df1 + df2) > 0 else np.nan
    else:
        eta_p2 = np.nan

    p_val = float(tab['Pr > F'].iloc[0]) if 'Pr > F' in tab.columns else np.nan
    F_val = float(tab['F Value'].iloc[0]) if 'F Value' in tab.columns else np.nan

    means = data.groupby(within)[val].mean().sort_index()

    prompt = (
        f"ANOVA Messwiederholung ({within}) auf {val}: F={F_val:.3f}, p={p_val:.3e}, "
        f"part. Eta²={eta_p2:.3f}, Mittelwerte={means.round(3).to_dict()}. "
        f"Erkläre knapp den Effekt und was das praktisch bedeutet. Keine Meta-Antworten!"
    )
    interp = sanitize_llm_output(get_llm_interpretation(prompt, model))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=means.index.astype(str), y=means.values, mode='lines+markers'))
    fig.update_layout(title="Messwiederholung: Mittelwerte", height=400, width=900)

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)
    html = (
        "<h4>ANOVA (Messwiederholung)</h4>"
        f"<p>F={F_val:.3f}, p={p_val:.3e}, part. Eta²={eta_p2:.3f}</p>"
        f"<div style='max-height:220px; overflow:auto;'>{tab.reset_index().to_html(index=False, border=0)}</div>"
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"
    )
    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/kruskal', methods=['POST'])
def kruskal():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p     = request.json or {}
    model = p.get('model', 'llama3.1p')
    grp   = p['group_col']
    val   = p['value_col']

    data = df[[grp, val]].dropna()
    grouped = [g[val].values for _, g in data.groupby(grp)]
    k = len(grouped)
    n = len(data)

    if k < 3:
        return jsonify(error="Kruskal-Wallis benötigt mind. 3 Gruppen"), 400

    h_stat, p_val = st.kruskal(*grouped)

    eps2 = (h_stat - k + 1) / (n - k) if (n - k) > 0 else np.nan

    prompt = (
        f"Kruskal-Wallis mit {k} Gruppen ({grp}): H={h_stat:.3f}, p={p_val:.3e}, "
        f"Epsilon²={eps2:.3f}. Erkläre kurz ob Gruppenunterschiede wahrscheinlich sind "
        f"und wie stark der Effekt ist. Keine Meta-Antworten!"
    )
    interp = sanitize_llm_output(get_llm_interpretation(prompt, model))

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Boxplot', 'Violinplot'))
    fig.add_trace(go.Box(y=data[val], x=data[grp], boxpoints='outliers'), row=1, col=1)
    fig.add_trace(go.Violin(y=data[val], x=data[grp], points='all', jitter=0.3), row=1, col=2)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, width=900, showlegend=False, title_text="Kruskal-Wallis")

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)
    html = (
        f"<h4>Kruskal-Wallis</h4>"
        f"<p>H={h_stat:.3f}, p={p_val:.3e}, Epsilon²={eps2:.3f}</p>"
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"
    )
    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/chi2', methods=['POST'])
def chi2():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    payload      = request.json or {}
    model        = payload.get('model', 'llama3.1p')
    var1         = payload.get('var1')
    var2         = payload.get('var2')
    top_n        = int(payload.get('top_n', 10))
    use_yates    = bool(payload.get('yates_correction', False))

    if var1 not in df.columns or var2 not in df.columns:
        return jsonify(error="Eine oder beide Variablen existieren nicht im Datensatz"), 400

    ct_full   = pd.crosstab(df[var1], df[var2]).sort_index()
    top_rows  = ct_full.sum(axis=1).nlargest(top_n).index
    top_cols  = ct_full.sum(axis=0).nlargest(top_n).index
    ct        = ct_full.loc[top_rows, top_cols]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return jsonify(error="Zu wenige Kategorien nach Filterung"), 400

    chi2_stat, p_val, dof, expected = st.chi2_contingency(ct, correction=use_yates)
    low_expected = (expected < 5).sum()
    warning = None
    if low_expected / expected.size > 0.2:
        warning = "Warnung: >20% der Zellen haben erwartete Häufigkeit <5; Ergebnisse könnten unzuverlässig sein."
    cramers_v = compute_cramers_v(chi2_stat, ct)
    effect_strength = (
        "schwach"  if cramers_v < 0.1 else
        "moderat"  if cramers_v < 0.3 else
        "stark"
    )

    warning_text = f"{warning} " if warning else ""
    prompt = (
        f'Chi-Quadrat-Test zwischen {escape(var1)} und {escape(var2)} '
        f'(Top-{top_n}): χ²={chi2_stat:.2f}, df={dof}, p={p_val:.4e}, '
        f"Cramér's V={cramers_v:.3f} ({effect_strength}). "
        f"{warning_text}"
        f"Formuliere in 2 Sätzen, ob ein signifikanter Zusammenhang besteht und was das praktisch bedeutet."
        f"Keine Meta-Antworten!"
    )
    try:
        raw_analysis = get_llm_interpretation(prompt, model)
        analysis     = sanitize_llm_output(raw_analysis)
    except:
        analysis = "Interpretation konnte nicht generiert werden."

    fig = px.imshow(
        ct,
        labels=dict(x=var2, y=var1, color="Anzahl"),
        x=ct.columns,
        y=ct.index,
        title=f"Kontingenz: {var1} vs. {var2} (Top {top_n})",
        color_continuous_scale="YlGnBu",
        aspect="auto",
        origin="lower"
    )

    text = [[str(v) for v in row] for row in ct.values] 

    mask = np.triu(np.ones_like(ct, dtype=bool), k=1)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if mask[i, j]:
                text[i][j] = ""

    fig.update_traces(text=text, texttemplate="%{text}")

    fig.update_xaxes(tickangle=45)
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=60))

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    html = (
        "<h4>Chi-Quadrat-Test</h4>"
        "<ul>"
        f"<li><strong>Chi²:</strong> {chi2_stat:.2f}</li>"
        f"<li><strong>p-Wert:</strong> {p_val:.4e}</li>"
        f"<li><strong>df:</strong> {dof}</li>"
        f"<li><strong>Cramér's V:</strong> {cramers_v:.3f} ({effect_strength})</li>"
        "</ul>"
        + (f"<p><em>{warning}</em></p>" if warning else "") +
        f"<div class='llm-analysis'><h3>Interpretation:</h3><p>{analysis}</p></div>"
    )

    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/fisher', methods=['POST'])
def fisher():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p     = request.json or {}
    model = p.get('model', 'llama3.1p')
    var1  = p.get('var1')
    var2  = p.get('var2')

    if var1 not in df.columns or var2 not in df.columns:
        return jsonify(error="Variable(n) existieren nicht"), 400

    ct = pd.crosstab(df[var1], df[var2]).dropna()
    if ct.shape != (2, 2):
        return jsonify(error="Fisher Exact ist nur für 2×2 Tabellen geeignet"), 400

    oddsratio, p_val = st.fisher_exact(ct.values)

    prompt = (
        f"Fisher Exact Test für {var1} vs {var2}: Odds Ratio={oddsratio:.3f}, p={p_val:.3e}. "
        f"Erkläre knapp, ob ein Zusammenhang wahrscheinlich ist und wie die Odds Ratio zu lesen ist. "
        f"Keine Meta-Antworten!"
    )
    interp = sanitize_llm_output(get_llm_interpretation(prompt, model))

    fig = px.imshow(
        ct,
        labels=dict(x=var2, y=var1, color="Anzahl"),
        x=ct.columns, y=ct.index,
        title=f"2×2 Kontingenz: {var1} vs {var2}",
        aspect="auto",
        origin="lower"
    )
    fig.update_traces(text=ct.values.astype(str), texttemplate="%{text}")
    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    html = (
        "<h4>Fisher Exact Test (2×2)</h4>"
        f"<p><strong>Odds Ratio:</strong> {oddsratio:.3f}</p>"
        f"<p><strong>p-Wert:</strong> {p_val:.3e}</p>"
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"
    )
    return jsonify(plotly_html=plotly_html, html=html)

@app.route('/cronbach_alpha', methods=['POST'])
def cronbach_alpha():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p     = request.json or {}
    model = p.get('model', 'llama3.1p')
    items = p.get('items', [])

    if not isinstance(items, list) or len(items) < 2:
        return jsonify(error="Bitte mindestens 2 Items übergeben"), 400

    X = df[items].dropna()
    if X.empty:
        return jsonify(error="Keine vollständigen Zeilen für die gewählten Items"), 400

    k = X.shape[1]
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else np.nan

    prompt = (
        f"Cronbachs Alpha für {items} (k={k}, n={len(X)}): alpha={alpha:.3f}. "
        "Erkläre kurz, ob die interne Konsistenz eher niedrig/mittel/hoch ist und was das praktisch heißt. Keine Meta-Antworten!"
    )
    interp = sanitize_llm_output(get_llm_interpretation(prompt, model))

    html = (
        "<h4>Cronbach's Alpha</h4>"
        f"<p><strong>Items:</strong> {', '.join(items)}</p>"
        f"<p><strong>n:</strong> {len(X)} | <strong>k:</strong> {k} | <strong>α:</strong> {alpha:.3f}</p>"
        f"<div class='llm-analysis'><h4>Interpretation vom LLM</h4><p>{interp}</p></div>"

    )
    return jsonify(html=html)

@app.route('/regression', methods=['POST'])
def regression():
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p = request.json
    model = p.get('model', 'llama3.1p')
    x_cols = p['x']         
    if not isinstance(x_cols, list) or len(x_cols) == 0:
        return jsonify(error="Keine X‑Variablen übergeben"), 400

    non_numeric = [c for c in x_cols if c not in df.select_dtypes(include=np.number).columns]
    if non_numeric:
        return jsonify(error=f"Nicht‑numerische X‑Spalten: {', '.join(non_numeric)}"), 400

    X = df[x_cols].dropna()

    if p['type'] == 'linear':
        y_col = p['y']
        if y_col not in df.select_dtypes(include=np.number):
            return jsonify(error="Y muss numerisch sein"), 400

        y      = df[y_col].loc[X.index].dropna()
        X_     = X.loc[y.index]
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_), index=X_.index, columns=X_.columns)
        poly   = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = pd.DataFrame(poly.fit_transform(X_scaled), index=X_.index, columns=poly.get_feature_names_out(X_.columns))
        X_sm   = sm.add_constant(X_poly)
        ols    = sm.OLS(y, X_sm).fit()

        influence   = ols.get_influence()
        cooks, _    = influence.cooks_distance
        std_resid   = influence.resid_studentized_internal
        leverage    = influence.hat_matrix_diag
        y_pred      = ols.predict(X_sm)
        r2          = ols.rsquared
        adj_r2      = ols.rsquared_adj
        mse         = mean_squared_error(y, y_pred)

        bp_test     = het_breuschpagan(ols.resid, X_sm)
        bp_labels   = ['LM Stat', 'p-value', 'F Stat', 'F p-value']
        bp_results  = dict(zip(bp_labels, bp_test))
        vif = pd.DataFrame({
            'feature': X_sm.columns,
            'VIF': [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
        })

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Residual-Plot', 'Q-Q Plot'))
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=y - y_pred,
                mode='markers',
                name='Residuen'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                line=dict(dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text='Vorhergesagt', row=1, col=1)
        fig.update_yaxes(title_text='Residuen',     row=1, col=1)

        osm, osr = st.probplot(y - y_pred, dist='norm')[0]
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Daten'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=osm, y=osm, mode='lines', name='45° Linie'),
            row=1, col=2
        )
        fig.update_xaxes(title_text='Theoretische Quantile', row=1, col=2)
        fig.update_yaxes(title_text='Sample Quantile',      row=1, col=2)

        fig.update_layout(
            height=500, width=1000,
            title_text=f"Diagnostische Plots für {y_col}",
            showlegend=False
        )

        plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        html = f"<h4>Diagnostische OLS-Regression: {y_col}</h4>"
        html += f"<p><strong>R²:</strong> {r2:.3f}, <strong>adj. R²:</strong> {adj_r2:.3f}, <strong>MSE:</strong> {mse:.3f}</p>"

        html += "<h4>Koeffizienten und Teststatistiken</h4><ul>"
        ci = ols.conf_int()
        for var in X_sm.columns:
            coef = ols.params[var]; pval = ols.pvalues[var]; ci_low, ci_high = ci.loc[var]
            if var == 'const':
                name = 'Intercept'
            else:
                name = var.replace('^2', '<sup>2</sup>').replace(' ', ' &times; ')
            html += (
                f"<li>{name}: β={coef:.4f} "
                f"(p={pval:.4f}, CI [{ci_low:.4f},{ci_high:.4f}])</li>"
            )
        html += "</ul>"

        html += "<h4>Breusch-Pagan-Test</h4><ul>"
        html += f"<li>LM Statistic: {bp_results['LM Stat']:.2f}, p={bp_results['p-value']:.4f}</li>"
        html += f"<li>F-Statistic: {bp_results['F Stat']:.2f}, p={bp_results['F p-value']:.4f}</li>"
        html += "</ul>"

        top_cooks = np.argsort(cooks)[-5:][::-1]
        html += "<h4>Top 5 Cook's Distance</h4><ul>"
        for idx in top_cooks:
            html += f"<li>Index {idx}: {cooks[idx]:.4f}, Leverage={leverage[idx]:.4f}, StdResid={std_resid[idx]:.4f}</li>"
        html += "</ul>"

        html += "<h4>VIF</h4><ul>"
        for _, row in vif.iterrows():
            feat = row['feature']
            if feat == 'const':
                name = 'Intercept'
            else:
                name = feat.replace('^2', '<sup>2</sup>').replace(' ', ' &times; ')
            html += f"<li>{name}: VIF={row['VIF']:.2f}</li>"
        html += "</ul>"

        prompt = (
            f"OLS-Diagnose für {y_col}: R²={r2:.2f}, adj.R²={adj_r2:.2f}, MSE={mse:.2f}. "
            f"Gib Hinweise auf Modellannahmen, Heteroskedastizität und Influential Observations."
            f"Keine Meta-Antworten!"
        )

        raw = get_llm_interpretation(prompt, model)
        analysis = sanitize_llm_output(raw)
        html += f"<div class='llm-analysis'><h3>Interpretation:</h3><p>{analysis}</p></div>"
        return jsonify(plotly_html=plotly_html, html=html)

    elif p['type'] == 'logistic':
        y_col = p['y']
        y = df[y_col].loc[X.index].dropna()
        if y.nunique() != 2:
            return jsonify(error="Für die logistische Regression muss Y binär sein"), 400

        X_ = X.loc[y.index].copy()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_),
            index=X_.index,
            columns=X_.columns
        )
        X_sm = sm.add_constant(X_scaled)
        logit = sm.Logit(y, X_sm).fit(disp=False)

        vif_data = pd.DataFrame({
            'feature': X_sm.columns,
            'VIF': [variance_inflation_factor(X_sm.values, i)
                    for i in range(X_sm.shape[1])]
        })
        y_proba = logit.predict(X_sm)
        auc      = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        y_pred = (y_proba >= 0.5).astype(int)
        acc    = accuracy_score(y, y_pred)
        class_rep = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"ROC Curve (AUC={auc:.2f})", "Kalibrierungs-Plot")
        )

        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False),
            row=1, col=1
        )
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate",   row=1, col=1)

        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        fig.add_trace(
            go.Scatter(x=prob_pred, y=prob_true, mode='markers+lines', name='Kalibrierung'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Mean predicted probability", row=1, col=2)
        fig.update_yaxes(title_text="Fraction of positives",      row=1, col=2)

        fig.update_layout(
            height=500, width=1000,
            title_text=f"Logistische Regression: {y_col}",
            showlegend=False
        )

        plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)
        plotly_html = plotly_html.replace(
            'style="height:500px; width:1000px;"',
            'style="margin:0; padding:0;"'
        )

        html = ""
        html += plotly_html  
        html += "<div class='stats'><ul>"
        html += f"<li>McFadden's Pseudo-R²: {logit.prsquared:.4f}</li>"
        ci = logit.conf_int()
        for var in X_sm.columns:
            coef = logit.params[var]; pval = logit.pvalues[var]
            ci_low, ci_high = ci.loc[var]
            name = 'Intercept' if var=='const' else var
            html += f"<li>{name}: β={coef:.4f} (p={pval:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])</li>"
        html += "</ul></div>"

        cm_html = '<table><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>'
        cm_html += f"<tr><th>True 0</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td></tr>"
        cm_html += f"<tr><th>True 1</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td></tr></table>"
        html += f"<h4>Konfusionsmatrix</h4>{cm_html}"

        html += "<h4>Classification Report</h4><ul>"
        for label, metrics in class_rep.items():
            if label in ['0','1']:
                html += (
                    f"<li>Klasse {label}: Präzision={metrics['precision']:.2f}, "
                    f"Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}</li>"
                )
        html += "</ul>"

        prompt = (
            f"Logistische Regression für {y_col} mit McFadden pseudo-R²={logit.prsquared:.2f}, "
            f"AUC={auc:.2f}, Accuracy={acc:.2%}. "
            f"Erkläre kurz die wichtigsten Kennzahlen und welche Features am stärksten sind."
            f"Keine Meta-Antworten!"
        )
        raw = get_llm_interpretation(prompt, model)
        analysis = sanitize_llm_output(raw)
        html += f"<div class='llm-analysis'><h3>Interpretation:</h3><p>{analysis}</p></div>"
        return jsonify(plotly_html=plotly_html, html=html)
        
    else:
        y_cols = p['y']
        if any(c not in df.select_dtypes(include=np.number).columns for c in y_cols):
            return jsonify(error="Alle Y-Variablen müssen numerisch sein"), 400

        Y = df[y_cols].dropna()
        common = X.index.intersection(Y.index)
        X_, Y_ = X.loc[common], Y.loc[common]

        X_sm = sm.add_constant(X_, prepend=True)
        models = {y: sm.OLS(Y_[y], X_sm).fit() for y in y_cols}

        mdl = LinearRegression().fit(X_, Y_)

        n_y, n_x = len(y_cols), len(x_cols)

        fig = make_subplots(
            rows=n_y, cols=n_x,
            subplot_titles=[f"{y} vs. {x}" for y in y_cols for x in x_cols],
            horizontal_spacing=0.15, 
            vertical_spacing=0.2
        )

        for i, y_var in enumerate(y_cols, start=1):
            for j, x_var in enumerate(x_cols, start=1):
                x_data = X_[x_var]
                y_data = Y_[y_var]
                coef    = mdl.coef_[y_cols.index(y_var)][x_cols.index(x_var)]
                intercept = mdl.intercept_[y_cols.index(y_var)]

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name=f"{y_var}~{x_var}"
                    ),
                    row=i, col=j
                )

                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                y_line = intercept + coef * x_line
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        showlegend=False
                    ),
                    row=i, col=j
                )

                fig.update_xaxes(title_text=x_var, row=i, col=j)
                fig.update_yaxes(title_text=y_var, row=i, col=j)

        for i in range(1, n_y+1):
            for j in range(1, n_x+1):
                fig.update_xaxes(
                    title_text=x_cols[j-1],
                    automargin=True,
                    tickfont=dict(size=12, color="black"),
                    row=i, col=j
                )
                fig.update_yaxes(
                    title_text=y_cols[i-1],
                    automargin=True,
                    tickfont=dict(size=12, color="black"),
                    row=i, col=j
                )

        fig.update_layout(
            height=300 * n_y,
            width=300 * n_x,
            title_text="Multivariate Regression Scatterplots mit Linien",
            showlegend=False
        )

        plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

        html = "<div class='stats'>"
        for y_var, mdl_sm in models.items():
            html += f"<h4>{y_var}</h4><ul>"
            html += f"<li>R²: {mdl_sm.rsquared:.4f}</li>"
            html += f"<li>adj. R²: {mdl_sm.rsquared_adj:.4f}</li>"
            ci = mdl_sm.conf_int()
            for x_var in ['const'] + x_cols:
                coef_ = mdl_sm.params[x_var]
                pval  = mdl_sm.pvalues[x_var]
                low, high = ci.loc[x_var]
                name = 'Intercept' if x_var == 'const' else x_var
                html += f"<li>{name}: β={coef_:.4f} (p={pval:.4f}, 95% CI [{low:.4f}, {high:.4f}])</li>"
            html += "</ul>"
        html += "</div>"

        coef_txt = "; ".join(
            f"{y}: [" + ", ".join(f"{x}={models[y].params[x]:.2f}" for x in x_cols) +
            f"] (R²={models[y].rsquared:.2f})"
            for y in y_cols
        )
        prompt = (
            "Multivariate lineare Regression. "
            f"Koeffizienten & Gütemaße: {coef_txt}. "
            "Gib in zwei bis drei Sätzen an, welche Zusammenhänge auffallen und "
            "welche Y-Variable am besten erklärt wird."
            f"Keine Meta-Antworten!"
        )
        raw = get_llm_interpretation(prompt, model)
        analysis = sanitize_llm_output(raw)
        html += f"<div class=\"llm-analysis\"><h3>Interpretation:</h3><p>{analysis}</p></div>"
        return jsonify(plotly_html=plotly_html, html=html)

@app.route('/timeseries', methods=['POST'])
def timeseries():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    p          = request.json
    model      = p.get('model', 'llama3.1p')
    date_col   = p['date']
    value_col  = p['value']

    if date_col not in df.columns or value_col not in df.columns:
        return jsonify(error="Spaltenname nicht gefunden"), 400

    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])
    if data.empty:
        return jsonify(error="Keine gültigen Datums-/Wert-Kombinationen"), 400

    yearly = (
        data
        .groupby(data[date_col].dt.year)[value_col]
        .mean()
        .rename("avg")
        .reset_index()
    )
    overall_mean = yearly['avg'].mean()
    std          = yearly['avg'].std()
    var          = yearly['avg'].var()
    min_year, min_val = yearly.loc[yearly['avg'].idxmin()]
    max_year, max_val = yearly.loc[yearly['avg'].idxmax()]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=yearly[date_col],
        y=yearly['avg'],
        mode='lines+markers+text',
        text=[f"{v:.2f}" for v in yearly['avg']],
        textposition="top center",
        name="Jahresdurchschnitt"
    ))

    fig.add_trace(go.Scatter(
        x=yearly[date_col],
        y=[overall_mean]*len(yearly),
        mode='lines',
        line=dict(dash='dash', color='red'),
        name=f"Gesamtschnitt ({overall_mean:.2f})"
    ))

    fig.update_layout(
        title=f"{value_col} – jährlicher Durchschnitt",
        xaxis_title="Jahr",
        yaxis_title=value_col,
        height=500,
        width=800,
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=True
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    prompt = (
        f"Zeitreihenanalyse für '{value_col}' (jährliche Durchschnittswerte).\n\n"
        f"Gesamtdurchschnitt: {overall_mean:.2f}\n"
        f"Minimum: {min_val:.2f} (Jahr {int(min_year)})\n"
        f"Maximum: {max_val:.2f} (Jahr {int(max_year)})\n"
        f"Standardabweichung: {std:.2f}\n\n"
        "Erkläre kurz:\n"
        "1. die Variabilität,\n"
        "2. auffällige Jahre bzw. Ausreißer,\n"
        "3. ob ein Trend erkennbar ist,\n"
        "4. und ziehe ein kurzes Fazit (1–2 Sätze)."
        "Keine Meta-Antworten!"
    )
    raw = get_llm_interpretation(prompt, model)
    interpretation = sanitize_llm_output(raw)

    stats_html = (
        "<h4>Key Statistics</h4>"
        "<ul>"
        f"<li><strong>Gesamtdurchschnitt:</strong> {overall_mean:.2f}</li>"
        f"<li><strong>Standardabweichung:</strong> {std:.2f}</li>"
        f"<li><strong>Varianz:</strong> {var:.2f}</li>"
        f"<li><strong>Minimum:</strong> {min_val:.2f} (Jahr {int(min_year)})</li>"
        f"<li><strong>Maximum:</strong> {max_val:.2f} (Jahr {int(max_year)})</li>"
        "</ul>"
    )

    html = (
        stats_html +
        f"<div class='llm-analysis'><h3>Interpretation:</h3><p>{interpretation}</p></div>"
    )

    combined = plotly_html + html
    return jsonify(html=combined)

@app.route('/cluster', methods=['POST'])
def cluster():
    global df
    if df is None:
        return jsonify(error="Keine Daten geladen"), 400

    params = request.get_json()  
    model_name = params.get('model', 'llama3.1p')
    vars_      = params.get('vars', [])
    if not vars_:
        return jsonify(error="Keine Variablen gewählt"), 400

    X = df[vars_].dropna()
    if X.empty:
        return jsonify(error="Keine Datenzeilen ohne NaN"), 400

    scaler = RobustScaler() if params.get('scaler') == 'robust' else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if params.get('remove_outliers', False):
        keep_mask      = detect_outliers(X_scaled, method='lof', n_neighbors=20)
        X_cluster      = X_scaled[keep_mask]
        outlier_mask   = ~keep_mask
    else:
        X_cluster    = X_scaled
        outlier_mask = np.zeros(X_scaled.shape[0], dtype=bool)

    method      = params.get('method')
    params_txt  = ""   
    sil_score   = None       

    if method == 'KMeans':
        k   = int(params.get('k', 3))
        mdl = KMeans(n_clusters=k, random_state=1).fit(X_cluster)
        labels_clean = mdl.labels_
        params_txt   = f"k={k}"

    elif method == 'Hierarchical':
        k   = int(params.get('k', 3))
        mdl = AgglomerativeClustering(n_clusters=k).fit(X_cluster)
        labels_clean = mdl.labels_
        params_txt   = f"k={k}"

    elif method == 'DBSCAN':
        eps  = float(params.get('eps', 0.5))
        mins = int(params.get('min_samples', 5))
        mdl  = DBSCAN(eps=eps, min_samples=mins).fit(X_scaled)
        labels = mdl.labels_     
        labels_clean = labels[~outlier_mask] if params.get('remove_outliers') else labels
        params_txt   = f"eps={eps}, min_samples={mins}"

    else:
        return jsonify(error="Unbekannte Methode"), 400

    if method in ('KMeans', 'Hierarchical'):
        labels = np.full(X_scaled.shape[0], -1, dtype=int)
        labels[~outlier_mask] = labels_clean  

    if method in ('KMeans', 'Hierarchical') and len(set(labels_clean)) > 1:
        sil_score  = silhouette_score(X_cluster, labels_clean)
        params_txt += f", silhouette={sil_score:.2f}"

    if params.get('projection') == 'tsne':
        coords = TSNE(n_components=2, random_state=1).fit_transform(X_scaled)
        proj_name = "t-SNE"
    else:
        coords = PCA(n_components=2).fit_transform(X_scaled)
        proj_name = "PCA"

    df_coords = pd.DataFrame(coords, columns=['Dim1', 'Dim2'])
    df_coords['cluster'] = labels.astype(str)

    fig = px.scatter(
        df_coords, x="Dim1", y="Dim2",
        color="cluster", symbol="cluster",
        title=f"{method} – 2D-Projektion ({proj_name})",
        labels={'cluster': 'Cluster'},
        hover_data={'cluster': True}
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), legend_title_text="Cluster")

    plotly_html = fig.to_html(full_html=False, include_plotlyjs=False)

    uniq, counts  = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(uniq, counts))
    sil_txt       = f"{sil_score:.2f}" if sil_score is not None else "n/a"
    outliers      = cluster_sizes.get(-1, 0)

    prompt = (
        f"Clustering mit Methode {method}\n"
        f"Variablen: {vars_}\n"
        f"Parameter: {params_txt}\n"
        f"Silhouette-Score: {sil_txt}\n"
        f"Cluster-Größen: {cluster_sizes}\n"
        f"{'Ausreißer (Label -1): ' + str(outliers) if outliers else ''}\n\n"
        "1. Beurteile, ob der Silhouette-Score (≥ 0,5) eine klare Trennung zeigt.\n"
        "2. Kommentiere sehr kleine Cluster (< 5 %).\n"
        "3. Erkläre den Kontext der Ausreißer und schlage Nachforschungen vor.\n"
        "4. Keine Meta-Antworten."
    )

    raw_answer   = get_llm_interpretation(prompt, model_name)
    interpretation = sanitize_llm_output(raw_answer)

    html = (
        f"<h4>Clustering: {method}</h4>"
        f"<p><strong>Parameter:</strong> {params_txt}</p>"
        f"<p><strong>Clustergrößen:</strong> {cluster_sizes}</p>"
        f"<div class='llm-analysis'><h3>Interpretation:</h3><p>{interpretation}</p></div>"
    )

    return jsonify(plotly_html=plotly_html, html=html)

if __name__ == "__main__":
    app.run(debug=True)
