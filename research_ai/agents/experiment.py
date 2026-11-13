"""
agents/experiment.py
====================
Agent 3: AI Research Experiment Agent
Tagline: "Design. Analyze. Discover."

Capabilities:
- Designs rigorous experimental protocols (Hypotheses H0/H1, Independent/Dependent/Control Variables,
  Sample Size & Power Reasoning, Step-by-Step Study Protocol, Internal/External Validity Threats)
- Generates safe, reproducible Python analysis code tailored to user-uploaded CSV/Excel datasets
  (or clearly labeled `DEMO DATA – not real` when requested)
- Enforces code execution safety via `utils/code_runner.py` (shows code to the user BEFORE execution,
  requires explicit "Run" click in UI, runs in restricted subprocess with timeout, no network,
  temp directory isolation, and allow-list of scientific libraries)
- Interprets ONLY actually executed statistical outputs (never invents numbers, p-values, or R2)
- Produces a canonical Methods & Results draft + downloadable `.py` script + `requirements.txt` snippet
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from utils.code_runner import (
    build_default_analysis_script,
    execute_sandboxed_python,
    validate_code_safety,
)
from utils.llm_factory import invoke_llm_text


def generate_experimental_design(
    llm: BaseChatModel,
    research_question: str,
    field: str = "Agricultural & Environmental AI",
    study_type: str = "Empirical / Computational & Field Trial",
    available_resources: str = "Standard university compute & lab/field dataset",
    dataset_summary: str = "No dataset uploaded yet (Design-Only Mode)",
    literature_context: Optional[str] = None,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> str:
    """
    Generate a complete experimental design protocol (hypotheses, variables, controls,
    sample-size power reasoning, protocol, validity threats, and multiple-testing plan).
    """
    system_prompt = (
        "You are the **AI Research Experiment Agent** inside ResearchAI ('Design. Analyze. Discover.').\n"
        "You are a senior biostatistician, experimental methodologist, and machine learning scientist.\n"
        "STRICT RULE: Never invent or fabricate empirical results, p-values, or sample measurements. "
        "When designing an experiment prior to code execution, provide rigorous methodological specifications "
        "and clearly distinguish theoretical power calculations from empirical results."
    )

    lit_block = f"\nLiterature Review Context:\n{literature_context[:1400]}\n" if literature_context else ""

    user_prompt = f"""Design a rigorous, publication-grade **Experimental & Analytical Protocol** for:

- **Research Question / Objective**: {research_question}
- **Field / Domain**: {field}
- **Study Design Type**: {study_type}
- **Available Resources**: {available_resources}
- **Dataset Schema / Summary**: {dataset_summary}
{lit_block}

Structure the Experimental Design in Markdown with these exact sections:

## 1. Testable Hypotheses ($H_0$ and $H_1$)
- Formulate primary and secondary null ($H_0$) and alternative ($H_1$) hypotheses in precise statistical terms.

## 2. Operationalized Variables & Experimental Controls
- **Independent Variables / Predictors / Treatments** (units, levels, coding)
- **Dependent Variables / Primary & Secondary Endpoints**
- **Control Variables / Covariates & Blocking Factors**
- **Positive & Negative Controls / Baseline Benchmarks**

## 3. Sample Size & Statistical Power Reasoning
- Explain the required sample size rationale (e.g., $\\alpha = 0.05$, power $1 - \\beta = 0.80$, expected effect size Cohen's $d$ / $\\eta^2$ / $f^2$, degrees of freedom, and cross-validation split strategy).

## 4. Step-by-Step Study & Analytical Protocol
- Pre-processing, quality control, missing-data handling, normality & homoscedasticity assumption checks (Shapiro-Wilk, Levene's test), parametric vs. non-parametric decision tree, regression/ML modeling, and multiple-testing correction (Bonferroni / Benjamini-Hochberg FDR).

## 5. Threats to Validity & Mitigation Matrix
- **Internal Validity**, **External Validity**, **Construct Validity**, and **Statistical Conclusion Validity** with concrete mitigation strategies.
"""

    return invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )


def generate_custom_analysis_code(
    llm: BaseChatModel,
    df: pd.DataFrame,
    analysis_goal: str,
    is_demo: bool = False,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> str:
    """
    Ask the LLM to write a safe, self-contained Python analysis script for `dataset.csv`
    using ONLY the allowed scientific libraries. Falls back to `build_default_analysis_script(df)`
    if the LLM output fails static AST safety checks.
    """
    cols_info = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
    sample_rows = df.head(4).to_string(index=False)

    system_prompt = (
        "You are a Scientific Python Code Generator for the ResearchAI Experiment Agent.\n"
        "Write clean, reproducible Python 3 code that analyzes `dataset.csv` in the current working directory.\n\n"
        "STRICT SECURITY & SANDBOX RULES:\n"
        "1. ONLY import from this allow-list: `pandas`, `numpy`, `scipy`, `statsmodels`, `sklearn`, "
        "`matplotlib`, `seaborn`, `plotly`, `Bio` (biopython), `math`, `statistics`, `json`, `re`, `warnings`.\n"
        "2. NEVER import `os`, `sys`, `subprocess`, `socket`, `requests`, `pathlib`, `shutil`, or `pickle`.\n"
        "3. NEVER call `eval()`, `exec()`, `compile()`, `__import__()`, or `input()`.\n"
        "4. Read data ONLY from `pd.read_csv('dataset.csv')`.\n"
        "5. Save summary tables as `table_*.csv` (e.g., `table_descriptive_stats.csv`, `table_hypothesis_tests.csv`) "
        "and figures as `fig_*.png` using `plt.savefig('fig_1_analysis.png', bbox_inches='tight')` and `plt.close()`.\n"
        "6. Print clear statistical test results (test statistics, exact p-values, effect sizes, R-squared, CV scores) to stdout.\n"
        "7. Output ONLY valid Python code inside a ```python ... ``` block."
    )

    demo_note = "NOTE: This dataset is labeled 'DEMO DATA – not real'. Include a comment at the top noting this.\n" if is_demo else ""

    user_prompt = f"""{demo_note}Write a complete Python analysis script for `dataset.csv`:
- **Shape**: {df.shape[0]} rows x {df.shape[1]} columns
- **Columns & Types**: {', '.join(cols_info)}
- **First 4 Rows Preview**:
{sample_rows}

- **Researcher's Specific Analysis Goal**: {analysis_goal}

Ensure the script computes:
1. Descriptive statistics (`mean`, `std`, `median`, `IQR`) saved to `table_descriptive_stats.csv`
2. Assumption checks & hypothesis tests (ANOVA / t-test / Mann-Whitney / Kruskal-Wallis / Chi-square as appropriate) with effect sizes
3. Regression or Machine Learning cross-validation (`statsmodels` or `sklearn`)
4. At least 2 publication-quality figures saved to `fig_1_main.png` and `fig_2_secondary.png`
"""

    try:
        raw = invoke_llm_text(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=provider_name,
            base_url=base_url,
        )
        match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
        candidate_code = match.group(1).strip() if match else raw.strip()
        is_safe, _ = validate_code_safety(candidate_code)
        if is_safe and "read_csv" in candidate_code:
            return candidate_code
    except Exception:
        pass

    return build_default_analysis_script(df, is_demo=is_demo)


def interpret_executed_results(
    llm: BaseChatModel,
    research_question: str,
    executed_stdout: str,
    table_summaries: str,
    is_demo: bool = False,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> str:
    """
    Generate a canonical Methods & Empirical Results narrative strictly from the
    ACTUAL stdout and tables produced by `execute_sandboxed_python()`.
    """
    demo_banner = (
        "⚠️ **IMPORTANT LABEL: DEMO DATA – NOT REAL.** The statistics below were computed on synthetic demonstration data for workflow illustration only.\n\n"
        if is_demo
        else ""
    )

    system_prompt = (
        "You are the **AI Research Experiment Agent** writing the **Statistical Methods & Empirical Results** section.\n"
        "CRITICAL SCIENTIFIC INTEGRITY RULE:\n"
        "- Every single number, mean, standard deviation, test statistic ($F$, $t$, $H$, $R^2$), and $p$-value "
        "you report MUST come verbatim from the EXECUTED CODE OUTPUT provided below.\n"
        "- NEVER invent, extrapolate, or alter any numerical result.\n"
        "- Include assumption checks, effect sizes, multiple-testing considerations, and methodological limitations."
    )

    user_prompt = f"""Write a rigorous **Methods & Empirical Results Section Draft** based strictly on the executed analysis below:

- **Research Question**: {research_question}
- **Data Source Status**: {'DEMO DATA – not real (synthetic demonstration dataset)' if is_demo else 'User-Uploaded Empirical Dataset'}

### Executed Python Subprocess `stdout`:
```text
{executed_stdout[:8000]}
```

### Generated Summary Tables:
```text
{table_summaries[:4000]}
```

Structure your response in Markdown with:
1. **Statistical & Computational Methods Paragraph** (ready to paste into a manuscript)
2. **Empirical Findings & Hypothesis Test Results** (reporting exact numbers, test statistics, $p$-values, and effect sizes from `stdout`)
3. **Model Validation & Diagnostics**
4. **Methodological Limitations & Multiple-Testing Considerations**
"""

    narrative = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )
    return demo_banner + narrative


def run_bioinformatics_sequence_stats(fasta_or_raw_seq: str) -> Dict[str, Any]:
    """
    Direct Biopython helper for basic bioinformatics analysis (sequence length,
    GC content, molecular weight approximation, reverse complement, and translation).
    Computed directly via Biopython — zero hallucination.
    """
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction

    records = []
    raw = (fasta_or_raw_seq or "").strip()
    if not raw:
        return {"records": [], "summary_df": pd.DataFrame()}

    # Parse simple FASTA or single sequence
    entries: List[Tuple[str, str]] = []
    if raw.startswith(">"):
        cur_id = "Seq_1"
        cur_lines: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_lines:
                    entries.append((cur_id, "".join(cur_lines)))
                cur_id = line[1:].strip() or f"Seq_{len(entries) + 1}"
                cur_lines = []
            else:
                cur_lines.append(re.sub(r"[^A-Za-z]", "", line).upper())
        if cur_lines:
            entries.append((cur_id, "".join(cur_lines)))
    else:
        clean_seq = re.sub(r"[^A-Za-z]", "", raw).upper()
        if clean_seq:
            entries.append(("Input_Sequence_1", clean_seq))

    for seq_id, seq_str in entries:
        if not seq_str:
            continue
        s = Seq(seq_str)
        gc_pct = round(gc_fraction(s) * 100.0, 2)
        rev_comp = str(s.reverse_complement())
        trimmed_len = (len(s) // 3) * 3
        peptide = str(s[:trimmed_len].translate()) if trimmed_len >= 3 else "N/A (<3 bp)"
        records.append(
            {
                "Sequence ID": seq_id,
                "Length (bp/aa)": len(s),
                "GC Content (%)": gc_pct,
                "A Count": seq_str.count("A"),
                "T/U Count": seq_str.count("T") + seq_str.count("U"),
                "G Count": seq_str.count("G"),
                "C Count": seq_str.count("C"),
                "5'->3' Preview": seq_str[:36] + ("..." if len(seq_str) > 36 else ""),
                "Reverse Complement Preview": rev_comp[:36] + ("..." if len(rev_comp) > 36 else ""),
                "Translated Peptide Preview": peptide[:30] + ("..." if len(peptide) > 30 else ""),
            }
        )

    return {
        "records": records,
        "summary_df": pd.DataFrame(records),
    }
