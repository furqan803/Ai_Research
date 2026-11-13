"""
utils/code_runner.py
====================
Safe, Sandboxed Subprocess Code Runner for the ResearchAI Experiment Agent.

Safety Architecture:
1. AST Static Inspection BEFORE execution:
   - Blocks dangerous modules (`os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`,
     `http`, `shutil`, `pathlib`, `pickle`, `ctypes`, `multiprocessing`, `pty`, `resource`).
   - Blocks dangerous builtins (`eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`).
   - Enforces an explicit allow-list of scientific & bioinformatics libraries (`pandas`, `numpy`,
     `scipy`, `statsmodels`, `sklearn`, `matplotlib`, `seaborn`, `plotly`, `Bio`, `math`,
     `statistics`, `collections`, `itertools`, `json`, `csv`, `re`, `datetime`).
2. Isolated Temporary Working Directory:
   - The user's uploaded dataset (or explicitly labeled `DEMO DATA – not real`) is written
     to `dataset.csv` inside an ephemeral `tempfile.TemporaryDirectory()`.
   - A Python wrapper overrides `socket.socket` to block all network calls and restricts
     `builtins.open` so files can ONLY be read/written inside the temporary directory.
3. Subprocess Timeout:
   - Runs via `subprocess.run(..., timeout=timeout_sec)` using the active Python interpreter.
   - Captures `stdout`, `stderr`, generated CSV summary tables (`table_*.csv`), and
     generated figures (`fig_*.png`).
"""

from __future__ import annotations

import ast
import base64
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


ALLOWED_TOP_MODULES: Set[str] = {
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "Bio",
    "biopython",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "json",
    "csv",
    "re",
    "datetime",
    "typing",
    "warnings",
}

FORBIDDEN_MODULES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "urllib3",
    "http",
    "ftplib",
    "telnetlib",
    "shutil",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "pty",
    "fcntl",
    "resource",
    "signal",
    "importlib",
    "builtins",
    " inspect",
    "webbrowser",
}

FORBIDDEN_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "delattr",
}


def validate_code_safety(code: str) -> Tuple[bool, List[str]]:
    """
    Parse user/LLM Python code with `ast` and verify it adheres to the allow-list
    and contains no dangerous calls or attribute accesses.
    """
    issues: List[str] = []
    if not code or not code.strip():
        return False, ["Code is empty."]

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Python SyntaxError on line {e.lineno}: {e.msg}"]

    for node in ast.walk(tree):
        # 1. Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".")[0]
                if top_mod in FORBIDDEN_MODULES:
                    issues.append(f"Forbidden module import `{alias.name}` is blocked for security.")
                elif top_mod not in ALLOWED_TOP_MODULES:
                    issues.append(
                        f"Module `{alias.name}` is not in the scientific allow-list ({', '.join(sorted(ALLOWED_TOP_MODULES))})."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_mod = node.module.split(".")[0]
                if top_mod in FORBIDDEN_MODULES:
                    issues.append(f"Forbidden module import `from {node.module} ...` is blocked for security.")
                elif top_mod not in ALLOWED_TOP_MODULES:
                    issues.append(
                        f"Module `{node.module}` is not in the scientific allow-list."
                    )

        # 2. Check forbidden function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
                if fn_name in FORBIDDEN_CALLS:
                    issues.append(f"Forbidden builtin call `{fn_name}()` is blocked.")

        # 3. Block dunder attribute traversal (__subclasses__, __globals__, etc.)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in ("__init__", "__name__", "__doc__"):
                issues.append(f"Access to dunder attribute `{node.attr}` is restricted.")

    return (len(issues) == 0), issues


def build_default_analysis_script(df: pd.DataFrame, is_demo: bool = False) -> str:
    """
    Generate a clean, reproducible default Python analysis script tailored to the
    actual columns of `df` (or the wheat drought demo dataset).
    """
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(exclude=["number"]).columns)

    banner = (
        "# NOTE: Running on DEMO DATA – not real empirical measurements.\n"
        if is_demo
        else "# Running on Researcher-Uploaded Dataset (dataset.csv)\n"
    )

    script = f"""{banner}import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

# 1. Load dataset from current isolated working directory
df = pd.read_csv("dataset.csv")
print("=== DATASET SHAPE & OVERVIEW ===")
print(f"Rows: {{df.shape[0]}}, Columns: {{df.shape[1]}}")
print("Columns:", list(df.columns))
print()

# 2. Descriptive Statistics (saved to table_descriptive_stats.csv)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if num_cols:
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    desc["iqr"] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    desc["skewness"] = df[num_cols].skew()
    desc.round(4).to_csv("table_descriptive_stats.csv")
    print("=== DESCRIPTIVE STATISTICS (NUMERIC VARIABLES) ===")
    print(desc[["count", "mean", "std", "median", "iqr", "min", "max"]].round(4).to_string())
    print()

# 3. Normality & Hypothesis Testing + Effect Size
if cat_cols and num_cols:
    group_col = cat_cols[-1] if len(cat_cols) > 1 else cat_cols[0]
    target_col = num_cols[-2] if len(num_cols) >= 2 else num_cols[0]
    print(f"=== HYPOTHESIS TESTING: '{{target_col}}' across groups of '{{group_col}}' ===")
    groups = [g[target_col].dropna().values for _, g in df.groupby(group_col) if len(g[target_col].dropna()) > 1]
    group_Summary = df.groupby(group_col)[num_cols].agg(["count", "mean", "std"]).round(3)
    group_Summary.to_csv("table_group_comparisons.csv")

    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        kw_stat, kw_pval = stats.kruskal(*groups)
        # Eta-squared effect size
        grand_mean = df[target_col].mean()
        ss_total = ((df[target_col] - grand_mean) ** 2).sum()
        ss_between = sum(len(g) * ((g.mean() - grand_mean) ** 2) for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        print(f"One-Way ANOVA: F = {{f_stat:.4f}}, p = {{p_val:.6g}}, Eta-squared (effect size) = {{eta_sq:.4f}}")
        print(f"Kruskal-Wallis (non-parametric check): H = {{kw_stat:.4f}}, p = {{kw_pval:.6g}}")
        print()

# 4. Correlation Matrix & OLS Regression
if len(num_cols) >= 2:
    corr_df = df[num_cols].corr().round(4)
    corr_df.to_csv("table_correlation_matrix.csv")
    y_col = num_cols[-2] if len(num_cols) >= 2 else num_cols[-1]
    x_cols = [c for c in num_cols if c != y_col][:4]
    X = sm.add_constant(df[x_cols].dropna())
    y = df.loc[X.index, y_col]
    ols_model = sm.OLS(y, X).fit()
    print(f"=== OLS MULTIPLE REGRESSION (Dependent Variable: {{y_col}}) ===")
    print(f"R-squared: {{ols_model.rsquared:.4f}} | Adj R-squared: {{ols_model.rsquared_adj:.4f}} | F-stat p-value: {{ols_model.f_pvalue:.6g}}")
    coef_table = pd.DataFrame({{
        "Coef": ols_model.params,
        "Std.Err": ols_model.bse,
        "t": ols_model.tvalues,
        "p_value": ols_model.pvalues,
    }}).round(4)
    coef_table.to_csv("table_regression_coefficients.csv")
    print(coef_table.to_string())
    print()

    # 5. Machine Learning Cross-Validation Check (Random Forest)
    if len(X) >= 6:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_folds = min(5, len(X))
        cv_scores = cross_val_score(rf, df.loc[X.index, x_cols], y, cv=cv_folds, scoring="r2")
        rf.fit(df.loc[X.index, x_cols], y)
        imp = pd.Series(rf.feature_importances_, index=x_cols).sort_values(ascending=False).round(4)
        print(f"=== MACHINE LEARNING VALIDATION ({{cv_folds}}-Fold CV Random Forest) ===")
        print(f"Mean CV R2: {{cv_scores.mean():.4f}} +/- {{cv_scores.std():.4f}}")
        print("Feature Importances:", imp.to_dict())
        print()

# 6. Publication-Ready Figures
sns.set_theme(style="whitegrid", palette="deep")

if len(num_cols) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    if cat_cols:
        grp = cat_cols[-1] if len(cat_cols) > 1 else cat_cols[0]
        sns.boxplot(data=df, x=grp, y=y_col, ax=axes[0], palette="viridis")
        sns.stripplot(data=df, x=grp, y=y_col, ax=axes[0], color="black", alpha=0.6, size=6)
        axes[0].set_title(f"Distribution of {{y_col}} by {{grp}}")
        axes[0].tick_params(axis="x", rotation=20)
    else:
        sns.histplot(df[num_cols[0]], kde=True, ax=axes[0], color="#14b8a6")
        axes[0].set_title(f"Distribution of {{num_cols[0]}}")

    sns.regplot(data=df, x=x_cols[0], y=y_col, ax=axes[1], color="#3b82f6", scatter_kws={{"s": 45, "alpha": 0.75}})
    axes[1].set_title(f"Regression: {{y_col}} vs {{x_cols[0]}}")
    plt.tight_layout()
    plt.savefig("fig_1_distributions_and_regression.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Correlation Heatmap
    fig2, ax2 = plt.subplots(figsize=(7, 5.5), dpi=150)
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax2)
    ax2.set_title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("fig_2_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig2)

# 7. Biopython Reference Sequence Check (Bioinformatics Verification)
sample_seq = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
print("=== BIOPYTHON SEQUENCE SANITY CHECK ===")
print(f"Sample CDS Length: {{len(sample_seq)}} bp | GC Content: {{gc_fraction(sample_seq)*100:.2f}}% | Translated Peptide: {{sample_seq.translate()}}")
"""
    return script


def execute_sandboxed_python(
    code: str,
    df: Optional[pd.DataFrame] = None,
    fasta_text: Optional[str] = None,
    timeout_sec: int = 25,
) -> Dict[str, Any]:
    """
    Execute validated Python code inside an isolated temporary directory with network
    and external file access disabled.

    Returns
    -------
    dict with keys:
      - success: bool
      - stdout: str
      - stderr: str
      - tables: List[Dict[str, Any]]  (name, dataframe)
      - figures: List[Dict[str, Any]] (name, bytes)
      - validation_issues: List[str]
    """
    is_safe, issues = validate_code_safety(code)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Code blocked by static safety check:\n- " + "\n- ".join(issues),
            "tables": [],
            "figures": [],
            "validation_issues": issues,
        }

    with tempfile.TemporaryDirectory(prefix="research_ai_exec_") as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()

        # Write dataset.csv if provided
        if df is not None:
            df.to_csv(tmp_path / "dataset.csv", index=False)

        # Write sequences.fasta if provided
        if fasta_text and fasta_text.strip():
            (tmp_path / "sequences.fasta").write_text(fasta_text.strip(), encoding="utf-8")

        # Write the user's code to user_analysis.py
        user_script_path = tmp_path / "user_analysis.py"
        user_script_path.write_text(code, encoding="utf-8")

        # Create sandboxed wrapper runner that blocks network sockets & outside file paths
        runner_code = f"""import builtins
import os
import socket
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ALLOWED_ROOT = Path({repr(str(tmp_path))}).resolve()

# 1. Disable all network access inside the subprocess (preserve class inheritance for stdlib ssl/urllib)
class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Network access is disabled inside the ResearchAI safe code runner.")

def _blocked_connect(*args, **kwargs):
    raise RuntimeError("Network access is disabled inside the ResearchAI safe code runner.")

socket.socket = _BlockedSocket
socket.create_connection = _blocked_connect

# 2. Restrict file open() to the isolated temporary directory (plus read-only Python stdlib/site-packages)
_orig_open = builtins.open

def _safe_open(file, mode="r", *args, **kwargs):
    try:
        target = Path(file).resolve()
    except Exception:
        return _orig_open(file, mode, *args, **kwargs)

    is_write = any(m in mode for m in ("w", "a", "x", "+"))
    if is_write:
        if not str(target).startswith(str(ALLOWED_ROOT)):
            raise PermissionError(f"Write access outside sandbox temp folder is blocked: {{target}}")
    else:
        # Allow reading files inside ALLOWED_ROOT or installed Python packages (e.g. matplotlib fonts/styles)
        import sys
        allowed_read_prefixes = [str(ALLOWED_ROOT), sys.prefix, sys.base_prefix, "/usr/share/fonts", "/etc/fonts"]
        if not any(str(target).startswith(p) for p in allowed_read_prefixes):
            raise PermissionError(f"Read access outside sandbox temp folder is blocked: {{target}}")

    return _orig_open(file, mode, *args, **kwargs)

builtins.open = _safe_open

# Execute user script
with _orig_open({repr(str(user_script_path))}, "r", encoding="utf-8") as f:
    script_source = f.read()

exec(compile(script_source, "user_analysis.py", "exec"), {{"__name__": "__main__"}})
"""
        runner_path = tmp_path / "_sandbox_runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")

        clean_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "MPLCONFIGDIR": str(tmp_path),
            "PYTHONIOENCODING": "utf-8",
        }

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path)],
                cwd=str(tmp_path),
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_sec} seconds. Simplify the computation or reduce cross-validation folds.",
                "tables": [],
                "figures": [],
                "validation_issues": [],
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Subprocess execution failed: {e}",
                "tables": [],
                "figures": [],
                "validation_issues": [],
            }

        # Collect generated CSV tables (any *.csv except dataset.csv)
        tables: List[Dict[str, Any]] = []
        for csv_file in sorted(tmp_path.glob("*.csv")):
            if csv_file.name == "dataset.csv":
                continue
            try:
                t_df = pd.read_csv(csv_file)
                tables.append({"name": csv_file.name, "df": t_df})
            except Exception:
                pass

        # Collect generated PNG figures
        figures: List[Dict[str, Any]] = []
        for png_file in sorted(tmp_path.glob("*.png")):
            try:
                img_bytes = png_file.read_bytes()
                figures.append(
                    {
                        "name": png_file.name,
                        "bytes": img_bytes,
                        "b64": base64.b64encode(img_bytes).decode("ascii"),
                    }
                )
            except Exception:
                pass

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "tables": tables,
            "figures": figures,
            "validation_issues": [],
        }
