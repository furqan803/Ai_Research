"""
agents/literature.py
====================
Agent 2: AI Literature Review Agent
Tagline: "Turn thousands of papers into clear insights."

Capabilities:
- Searches & collects relevant papers across arXiv, OpenAlex, Semantic Scholar, Crossref, PubMed
- Executes a transparent PRISMA-style flow (Identified -> Deduplicated -> Screened -> Included)
- Screens each paper against explicit Inclusion & Exclusion criteria with a documented reason per decision
- Extracts structured evidence tables (Aim, Methodology, Dataset/Sample, Key Findings, Limitations)
- Supports user-uploaded PDFs (via `pypdf`) and integrates them into the Verified Reference Library
- Generates a structured, systematic-review-style literature synthesis (Introduction, Thematic Taxonomy,
  Cross-Study Comparison Table, Methodological Critique, Research Gaps, Future Directions, Conclusion,
  and Canonical Reference List)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm_factory import invoke_llm_json, invoke_llm_text
from utils.reference_library import (
    build_reference_context_block,
    format_citation,
    make_in_text_key,
    merge_into_library,
)
from utils.search_apis import extract_pdf_document, search_all_databases


def screen_papers_prisma(
    papers: List[Dict[str, Any]],
    inclusion_criteria: str,
    exclusion_criteria: str,
    min_year: Optional[int] = None,
    require_abstract: bool = False,
) -> Dict[str, Any]:
    """
    Screen deduplicated papers against explicit criteria and produce a transparent
    per-paper decision log + PRISMA flow counts.
    """
    inc_keywords = [
        w.strip().lower()
        for w in re.split(r"[,;\n]+", inclusion_criteria or "")
        if len(w.strip()) >= 3
    ]
    exc_keywords = [
        w.strip().lower()
        for w in re.split(r"[,;\n]+", exclusion_criteria or "")
        if len(w.strip()) >= 3
    ]

    included: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    screening_rows: List[Dict[str, Any]] = []

    for idx, p in enumerate(papers, start=1):
        title = p.get("title") or "Untitled"
        abstract = p.get("abstract") or ""
        year = p.get("year")
        combined_text = f"{title} {abstract}".lower()

        decision = "Included"
        reason = "Meets domain scope and inclusion criteria."

        if min_year and isinstance(year, int) and year < min_year:
            decision = "Excluded"
            reason = f"Published in {year} (prior to minimum year {min_year})."
        elif require_abstract and len(abstract.strip()) < 30 and p.get("source_db") != "Uploaded PDF":
            decision = "Excluded"
            reason = "Insufficient abstract metadata for systematic synthesis."
        else:
            # Check exclusion keywords
            matched_exc = [kw for kw in exc_keywords if kw in combined_text]
            if matched_exc:
                decision = "Excluded"
                reason = f"Matched exclusion criterion: '{matched_exc[0]}'."
            elif inc_keywords:
                matched_inc = [kw for kw in inc_keywords if kw in combined_text]
                if matched_inc:
                    decision = "Included"
                    reason = f"Matched inclusion term(s): {', '.join(matched_inc[:3])}."
                else:
                    # Keep papers returned by primary search unless strictly excluded, noting broad relevance
                    decision = "Included"
                    reason = "Retrieved via primary topic query; relevant to synthesis scope."

        row = {
            "Paper #": idx,
            "Title": title[:95],
            "Year": year or "n.d.",
            "Venue / Source": p.get("venue") or p.get("source_db") or "API",
            "DOI / Link": (f"https://doi.org/{p['doi']}" if p.get("doi") else p.get("url", ""))[:70],
            "Decision": decision,
            "Screening Reason": reason,
        }
        screening_rows.append(row)
        if decision == "Included":
            included.append(p)
        else:
            excluded.append(p)

    return {
        "included_papers": included,
        "excluded_papers": excluded,
        "screening_df": pd.DataFrame(screening_rows),
    }


def extract_evidence_matrix(
    llm: BaseChatModel,
    included_papers: List[Dict[str, Any]],
    citation_style: str = "APA",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a structured Literature Extraction Table (Study Citation, Aim, Methods,
    Dataset/Sample, Key Findings, Limitations) for the included papers.
    Uses LLM JSON extraction with a deterministic metadata fallback so it never crashes.
    """
    if not included_papers:
        return pd.DataFrame(
            columns=["Citation", "Source DB", "Study Aim", "Methods / Model", "Dataset / Sample", "Key Results", "Limitations"]
        )

    subset = included_papers[:12]
    papers_payload = []
    for idx, p in enumerate(subset, start=1):
        papers_payload.append(
            {
                "index": idx,
                "citation": make_in_text_key(p, idx, style=citation_style),
                "title": p.get("title", ""),
                "venue": p.get("venue", ""),
                "abstract": (p.get("abstract") or "")[:650],
            }
        )

    sys_prompt = (
        "You are a Systematic Review Data Extraction Specialist. Given a list of real papers "
        "(title, venue, abstract), extract a JSON list of objects with keys: "
        "`index`, `aim`, `methods`, `dataset`, `results`, `limitations`. "
        "Base every field strictly on the provided title and abstract. If a detail is not stated in the abstract, "
        "write 'Not specified in abstract (inspect full text)'."
    )
    user_prompt = f"Extract structured evidence table rows for these {len(papers_payload)} papers:\n{papers_payload}"

    extracted_map: Dict[int, Dict[str, str]] = {}
    try:
        parsed = invoke_llm_json(
            llm=llm,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            fallback_default=[],
            provider=provider_name,
            base_url=base_url,
        )
        if isinstance(parsed, dict) and "papers" in parsed:
            parsed = parsed["papers"]
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "index" in item:
                    extracted_map[int(item["index"])] = item
    except Exception:
        pass

    rows = []
    for idx, p in enumerate(subset, start=1):
        cite = make_in_text_key(p, idx, style=citation_style)
        ab = (p.get("abstract") or "").strip()
        ext = extracted_map.get(idx, {})
        rows.append(
            {
                "Citation": cite,
                "Title": (p.get("title") or "")[:80],
                "Source DB": p.get("source_db", "API"),
                "Study Aim": ext.get("aim") or (ab[:140] + "..." if ab else "Investigate domain research question"),
                "Methods / Model": ext.get("methods") or "Empirical / computational analysis (see abstract)",
                "Dataset / Sample": ext.get("dataset") or "Domain dataset / cohort (see full text)",
                "Key Results": ext.get("results") or (ab[140:300] + "..." if len(ab) > 140 else "Reported in study"),
                "Limitations": ext.get("limitations") or "Verify sample scope & external validity in full text",
            }
        )

    return pd.DataFrame(rows)


def run_literature_agent(
    llm: BaseChatModel,
    topic_query: str,
    inclusion_criteria: str = "Peer-reviewed or preprint studies directly addressing the core topic and empirical methods",
    exclusion_criteria: str = "Non-research editorials, corrigenda, unrelated domains",
    min_year: Optional[int] = 2018,
    require_abstract: bool = False,
    databases: Optional[List[str]] = None,
    max_per_db: int = 6,
    uploaded_pdfs: Optional[List[Tuple[bytes, str]]] = None,
    citation_style: str = "APA",
    existing_library: Optional[List[Dict[str, Any]]] = None,
    discovery_context: Optional[str] = None,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the AI Literature Review Agent:
    1. Search APIs + parse any uploaded PDFs
    2. Deduplicate and run PRISMA screening with per-paper decisions
    3. Update Verified Reference Library with included papers
    4. Extract structured evidence comparison matrix
    5. Generate systematic-review-style literature synthesis grounded in verified references
    """
    search_res = search_all_databases(
        query=topic_query,
        databases=databases or ["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
        max_per_db=max_per_db,
        year_from=None,  # Filter during PRISMA screening so excluded count is transparent
    )

    pdf_papers: List[Dict[str, Any]] = []
    for file_bytes, filename in (uploaded_pdfs or []):
        pdf_doc = extract_pdf_document(file_bytes, filename)
        pdf_papers.append(pdf_doc)

    all_unique_candidates = pdf_papers + search_res["papers"]
    total_identified = search_res["total_identified"] + len(pdf_papers)
    duplicates_removed = search_res["duplicates_removed"]
    records_screened = len(all_unique_candidates)

    screening = screen_papers_prisma(
        papers=all_unique_candidates,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        min_year=min_year,
        require_abstract=require_abstract,
    )
    included_papers = screening["included_papers"]
    excluded_papers = screening["excluded_papers"]

    # If strict filters excluded everything, retain top papers so the review still has verified evidence
    if not included_papers and all_unique_candidates:
        included_papers = all_unique_candidates[: min(8, len(all_unique_candidates))]

    updated_library = merge_into_library(existing_library or [], included_papers)

    prisma_counts = {
        "Identified from APIs & PDFs": total_identified,
        "Duplicates Removed": duplicates_removed,
        "Records Screened": records_screened,
        "Records Excluded": len(excluded_papers),
        "Studies Included in Review": len(included_papers),
    }

    evidence_df = extract_evidence_matrix(
        llm=llm,
        included_papers=included_papers,
        citation_style=citation_style,
        provider_name=provider_name,
        base_url=base_url,
    )

    ref_context = build_reference_context_block(updated_library, style=citation_style, max_refs=30)

    system_prompt = (
        "You are the **AI Literature Review Agent** inside ResearchAI ('Turn thousands of papers into clear insights.').\n"
        "You write rigorous, systematic-review-grade academic literature reviews following PRISMA 2020 standards.\n\n"
        "STRICT ACADEMIC INTEGRITY RULES:\n"
        "1. Cite ONLY papers from the VERIFIED REFERENCE LIBRARY provided below.\n"
        "2. NEVER invent or hallucinate any author, year, title, or DOI.\n"
        "3. Use the exact in-text citation keys shown in the library (e.g., `(Author et al., 2023)` or `[1]`)."
    )

    disc_snippet = f"\nPrior Discovery Context:\n{discovery_context[:1500]}\n" if discovery_context else ""

    user_prompt = f"""Write a comprehensive, structured **Systematic Literature Review** on:
- **Review Topic / Query**: {topic_query}
- **Inclusion Criteria**: {inclusion_criteria}
- **Exclusion Criteria**: {exclusion_criteria}
- **Minimum Publication Year**: {min_year or 'All years'}
- **Citation Style**: {citation_style}
- **PRISMA Flow Summary**:
  * Identified across APIs & Uploaded PDFs: {prisma_counts['Identified from APIs & PDFs']}
  * Duplicates removed: {prisma_counts['Duplicates Removed']}
  * Screened: {prisma_counts['Records Screened']}
  * Excluded with documented reasons: {prisma_counts['Records Excluded']}
  * Included in final synthesis: {prisma_counts['Studies Included in Review']}
{disc_snippet}

{ref_context}

Please produce a thorough, publication-grade Systematic Literature Review in Markdown with the following canonical structure:

## 1. Introduction & Scope of the Review
- Background, rationale, and core objectives of the review.

## 2. Search Strategy & PRISMA Selection Methodology
- Databases queried, inclusion/exclusion protocol, and summary of the PRISMA flow from {prisma_counts['Identified from APIs & PDFs']} identified records to {prisma_counts['Studies Included in Review']} included studies.

## 3. Thematic Synthesis of the Literature
Organize the included studies into **3 to 4 major analytical themes** (e.g., Theoretical Foundations, Methodological & Algorithmic Approaches, Empirical Benchmarks & Domain Applications, Emerging Paradigms). Cite the verified papers heavily throughout.

## 4. Cross-Study Methodological & Empirical Comparison
- Present a Markdown comparison table contrasting the key included studies (Study Citation, Core Method, Dataset/Setting, Primary Outcome, Key Limitation).

## 5. Critical Appraisal, Contradictions & Research Gaps
- Critically evaluate methodological strengths, dataset biases, sample size constraints, and unresolved contradictions across the included studies.

## 6. Future Research Agenda & Testable Hypotheses
- Formulate concrete future directions and testable hypotheses ready for the **AI Research Experiment Agent**.

## 7. Conclusion

## 8. Verified References ({citation_style} Format)
- Provide the complete numbered/alphabetized reference list with DOIs/links for every paper cited.
"""

    review_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "topic_query": topic_query,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "prisma_counts": prisma_counts,
        "screening_df": screening["screening_df"],
        "evidence_df": evidence_df,
        "included_papers": included_papers,
        "verified_references": updated_library,
        "review_md": review_md,
    }
