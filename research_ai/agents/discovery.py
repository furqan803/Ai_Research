"""
agents/discovery.py
===================
Agent 1: AI Research Discovery Agent
Tagline: "Find what matters. Discover new possibilities."

Capabilities:
- Scans top scholarly databases (arXiv, Semantic Scholar, OpenAlex, Crossref, PubMed)
- Computes real bibliometric trends from retrieved papers (publication counts per year,
  top authors, top venues) for interactive Plotly/Matplotlib visualization
- Identifies research gaps (underexplored combinations, contradictions, missing populations/methods)
  grounded strictly in the retrieved literature
- Suggests 5 novel research ideas, each with:
  * Working Title
  * Draft Research Question (PICO / FINER aligned)
  * Novelty Check against retrieved papers
  * Feasibility & Resource Fit
  * Expected Scientific & Societal Impact
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm_factory import invoke_llm_text
from utils.reference_library import build_reference_context_block, merge_into_library
from utils.search_apis import search_all_databases


def compute_bibliometric_trends(papers: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Compute real publication counts per year, top authors, and top venues from
    the retrieved API papers.
    """
    year_counts: Counter = Counter()
    author_counts: Counter = Counter()
    venue_counts: Counter = Counter()
    db_counts: Counter = Counter()

    for p in papers:
        yr = p.get("year")
        if isinstance(yr, int) and 1980 <= yr <= 2030:
            year_counts[yr] += 1
        for au in (p.get("authors") or []):
            if au and len(au.strip()) > 1:
                author_counts[au.strip()] += 1
        venue = (p.get("venue") or "").strip()
        if venue:
            venue_counts[venue[:60]] += 1
        db = (p.get("source_db") or "API").strip()
        db_counts[db] += 1

    df_years = (
        pd.DataFrame(sorted(year_counts.items()), columns=["Year", "Publications"])
        if year_counts
        else pd.DataFrame(columns=["Year", "Publications"])
    )
    df_authors = (
        pd.DataFrame(author_counts.most_common(10), columns=["Author", "Paper Count"])
        if author_counts
        else pd.DataFrame(columns=["Author", "Paper Count"])
    )
    df_venues = (
        pd.DataFrame(venue_counts.most_common(10), columns=["Venue / Journal", "Paper Count"])
        if venue_counts
        else pd.DataFrame(columns=["Venue / Journal", "Paper Count"])
    )
    df_dbs = (
        pd.DataFrame(db_counts.most_common(), columns=["Database", "Papers Retrieved"])
        if db_counts
        else pd.DataFrame(columns=["Database", "Papers Retrieved"])
    )

    return {
        "years": df_years,
        "authors": df_authors,
        "venues": df_venues,
        "databases": df_dbs,
    }


def run_discovery_agent(
    llm: BaseChatModel,
    field: str,
    keywords: str,
    career_stage: str = "PhD Student",
    available_resources: str = "Standard university compute & public datasets",
    preferred_depth: str = "Deep PhD-Level Exploration",
    databases: Optional[List[str]] = None,
    max_per_db: int = 6,
    citation_style: str = "APA",
    existing_library: Optional[List[Dict[str, Any]]] = None,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the AI Research Discovery Agent end-to-end:
    1. Search live academic databases
    2. Merge into Verified Reference Library
    3. Compute real bibliometric distributions
    4. Synthesize research trends, evidence-backed gaps, and 5 novel research ideas
    """
    query = f"{field} {keywords}".strip()
    search_res = search_all_databases(
        query=query,
        databases=databases or ["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
        max_per_db=max_per_db,
    )
    retrieved_papers = search_res["papers"]
    updated_library = merge_into_library(existing_library or [], retrieved_papers)
    trends = compute_bibliometric_trends(retrieved_papers)

    ref_context = build_reference_context_block(updated_library, style=citation_style, max_refs=25)

    system_prompt = (
        "You are the **AI Research Discovery Agent** inside ResearchAI ('Find what matters. Discover new possibilities.').\n"
        "Your mission is to guide BS students, MS/PhD candidates, professors, and R&D scientists from raw curiosity "
        "to high-impact, defensible research directions.\n\n"
        "STRICT ACADEMIC INTEGRITY RULES:\n"
        "1. NEVER fabricate citations, DOIs, author names, or publication years.\n"
        "2. Cite ONLY papers provided in the VERIFIED REFERENCE LIBRARY below using their exact in-text citation keys.\n"
        "3. If no papers were returned by the APIs for a sub-topic, explicitly state that no verified papers were found rather than inventing one."
    )

    user_prompt = f"""Perform a comprehensive **Research Discovery & Gap Analysis** with the following researcher parameters:

- **Field / Domain**: {field}
- **Specific Keywords / Topic**: {keywords}
- **Career Stage**: {career_stage}
- **Available Resources & Infrastructure**: {available_resources}
- **Analysis Depth**: {preferred_depth}
- **Preferred Citation Style**: {citation_style}
- **Total Papers Retrieved Across Databases**: {len(retrieved_papers)} (Total identified before deduplication: {search_res['total_identified']})

{ref_context}

Please structure your report in clear, well-organized Markdown with the following sections:

## 1. Executive Landscape & Emerging Research Trends
- Synthesize the dominant themes, methodological shifts, and temporal evolution evident across the retrieved papers.
- Cite specific verified papers from the library above to anchor every trend claim.

## 2. Evidence-Backed Research Gaps
Identify at least **4 concrete research gaps** supported by the retrieved literature:
- **Underexplored Combinations** (e.g., methods/modalities rarely combined in this domain)
- **Empirical Contradictions or Mixed Findings** across existing studies
- **Missing Populations, Environments, or Edge Cases**
- **Methodological / Scalability Bottlenecks**
*(Provide direct evidence from the retrieved papers for each gap.)*

## 3. Five (5) Novel Research Ideas
Generate **5 distinct, high-impact research ideas** tailored to a **{career_stage}** with **{available_resources}**.
For EACH of the 5 ideas, provide a structured card/subsection with:
1. **Working Title** (canonical academic title)
2. **Draft Research Question** (precise, testable, and operationalized)
3. **Core Hypothesis & Proposed Approach**
4. **Novelty Check Against Retrieved Literature** (explicitly contrast against the verified papers above)
5. **Feasibility & Resource Requirements** (why it fits {career_stage} and {available_resources})
6. **Expected Scientific & Real-World Impact**

## 4. Recommended Immediate Next Step for Literature Screening
- Suggest the #1 most promising idea to carry forward into the **AI Literature Review Agent** and specify recommended inclusion/exclusion criteria.

## 5. Verified Citations Used
- List all verified references cited in this discovery report in **{citation_style}** format with their DOIs/URLs.
"""

    report_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "query": query,
        "field": field,
        "keywords": keywords,
        "career_stage": career_stage,
        "available_resources": available_resources,
        "preferred_depth": preferred_depth,
        "search_summary": search_res,
        "papers": retrieved_papers,
        "verified_references": updated_library,
        "trends": trends,
        "report_md": report_md,
    }
