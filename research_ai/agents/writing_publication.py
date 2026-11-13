"""
agents/writing_publication.py
=============================
Agent 4: AI Academic Writing & Publication Agent
Tagline: "From data to publication."

Capabilities:
- Writes full, canonically structured academic manuscripts (Title, Structured Abstract, Keywords,
  Highlights, 1. Introduction, 2. Related Work / Literature Synthesis, 3. Materials & Methods,
  4. Empirical Results, 5. Discussion, 6. Limitations & Ethical Considerations, 7. Conclusion,
  Acknowledgments, AI-Use Disclosure, CRediT Authorship Statement, Data Availability, and
  Canonical Verified Reference List)
- Uses ONLY the Literature Agent's verified references and the Experiment Agent's actual executed results
- Recommends suitable peer-reviewed journals via live Crossref / OpenAlex Sources APIs with scope-fit
  reasoning, open-access status, DOAJ status, and links to author guidelines (never invents Impact Factors
  or acceptance rates; always reminds user to verify on the journal website)
- Simulates realistic Peer Review Feedback (Reviewer 1, Reviewer 2, and Associate Editor Decision with
  Major/Minor comments)
- Generates a Point-by-Point Response-to-Reviewers Letter & Revision Action Plan
- Generates Cover Letter, 5 Bullet Highlights, Keywords, and a Predatory-Journal Warning Checklist
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm_factory import invoke_llm_text
from utils.reference_library import (
    build_reference_context_block,
    format_full_bibliography,
)
from utils.search_apis import search_journals


PREDATORY_JOURNAL_CHECKLIST: List[Dict[str, str]] = [
    {
        "Check Item": "1. DOAJ & Major Indexing Verification",
        "What to Verify": "Confirm whether an Open Access journal is listed in the Directory of Open Access Journals (https://doaj.org), Scopus, Web of Science (Clarivate Master Journal List), or PubMed/MEDLINE.",
        "Red Flag": "Claims fake metrics like 'Global Impact Factor', 'Universal Impact Factor', or non-Clarivate 'JIF' badges.",
    },
    {
        "Check Item": "2. Publisher & ISSN Transparency",
        "What to Verify": "Verify the ISSN on the official ISSN Portal (https://portal.issn.org) and check membership in COPE (Committee on Publication Ethics) or OASPA.",
        "Red Flag": "Mismatched ISSNs, cloned journal titles mimicking established Nature/IEEE/Elsevier journals, or hidden physical address.",
    },
    {
        "Check Item": "3. Peer Review Timeline & Rigor",
        "What to Verify": "Ensure the journal describes a transparent, multi-reviewer peer-review process with realistic timelines (typically 4–12+ weeks).",
        "Red Flag": "Promises 'guaranteed acceptance within 48–72 hours' or aggressive unsolicited spam emails praising unrelated papers.",
    },
    {
        "Check Item": "4. Editorial Board Authenticity",
        "What to Verify": "Check 3–5 editorial board members' institutional university pages to confirm they actually serve as editors for the journal.",
        "Red Flag": "Scholars listed without their permission, missing institutional affiliations, or generic Gmail/Yahoo editorial contacts.",
    },
    {
        "Check Item": "5. Article Processing Charges (APCs) & Copyright",
        "What to Verify": "APCs, waiver policies, and Creative Commons licenses (e.g., CC BY 4.0) must be clearly stated BEFORE submission.",
        "Red Flag": "Hidden surprise fees demanded only after acceptance, or demanding copyright transfer while charging an Open Access fee.",
    },
]


def recommend_journals_with_fit(
    llm: BaseChatModel,
    topic: str,
    field: str,
    target_journal_hint: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve real journals from OpenAlex Sources & Crossref Journals, then generate
    scope-fit reasoning without fabricating Impact Factors or acceptance rates.
    """
    query = f"{field} {topic}".strip()
    api_journals = search_journals(query, max_results=8)

    # Supplement with well-known flagship publishers if API returned very few
    if len(api_journals) < 3:
        api_journals.extend(
            [
                {
                    "name": "Nature Communications",
                    "publisher": "Springer Nature",
                    "issn": "2041-1723",
                    "is_open_access": True,
                    "in_doaj": True,
                    "homepage_url": "https://www.nature.com/ncomms/submit/author-instructions",
                    "works_count": None,
                    "cited_by_count": None,
                    "two_yr_mean_citedness": None,
                    "h_index": None,
                    "source_api": "Curated Major Venue",
                    "metrics_note": "Verify current Journal Impact Factor and APCs on the journal website.",
                },
                {
                    "name": "IEEE Access",
                    "publisher": "Institute of Electrical and Electronics Engineers (IEEE)",
                    "issn": "2169-3536",
                    "is_open_access": True,
                    "in_doaj": True,
                    "homepage_url": "https://ieeeaccess.ieee.org/guide-for-authors/",
                    "works_count": None,
                    "cited_by_count": None,
                    "two_yr_mean_citedness": None,
                    "h_index": None,
                    "source_api": "Curated Major Venue",
                    "metrics_note": "Verify current Journal Impact Factor and author guidelines on the journal website.",
                },
                {
                    "name": "PLOS ONE",
                    "publisher": "Public Library of Science",
                    "issn": "1932-6203",
                    "is_open_access": True,
                    "in_doaj": True,
                    "homepage_url": "https://journals.plos.org/plosone/s/submission-guidelines",
                    "works_count": None,
                    "cited_by_count": None,
                    "two_yr_mean_citedness": None,
                    "h_index": None,
                    "source_api": "Curated Major Venue",
                    "metrics_note": "Verify current metrics and submission guidelines on the journal website.",
                },
            ]
        )

    rows = []
    for j in api_journals:
        rows.append(
            {
                "Journal Title": j["name"],
                "Publisher": j["publisher"],
                "ISSN": j["issn"] or "Verify on site",
                "Open Access": "Yes" if j["is_open_access"] else "Hybrid / Subscription",
                "DOAJ Listed": "Yes" if j["in_doaj"] else "Verify on DOAJ",
                "OpenAlex 2yr Citedness": j["two_yr_mean_citedness"] if j["two_yr_mean_citedness"] is not None else "Verify on site",
                "OpenAlex h-index": j["h_index"] if j["h_index"] is not None else "Verify on site",
                "Official Link / Guidelines": j["homepage_url"],
                "Metrics Verification Note": j["metrics_note"],
            }
        )
    journals_df = pd.DataFrame(rows)

    system_prompt = (
        "You are an Academic Publication Strategist.\n"
        "STRICT RULE: NEVER invent or state unverified Journal Impact Factors, CiteScores, acceptance rates, "
        "or review turnaround times. Always instruct the author to 'verify current Impact Factor, acceptance rate, "
        "and APCs on the official journal website'."
    )
    user_prompt = f"""Analyze scope fit for the following manuscript topic and retrieved journals:
- **Manuscript Topic**: {topic}
- **Field**: {field}
- **Author's Preferred Venue Hint**: {target_journal_hint or 'Open to best fit'}
- **Retrieved Journals from OpenAlex / Crossref**:
{journals_df.to_string(index=False)}

Provide:
1. **Scope-Fit & Tier Recommendation Analysis** for the top 5 journals above (why the topic fits, target readership, and link to verify guidelines).
2. **Pre-Submission Formatting & Compliance Note** (reminder to verify Impact Factor, acceptance rate, and formatting templates directly on the journal website).
"""
    analysis_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "journals_df": journals_df,
        "journal_analysis_md": analysis_md,
        "predatory_checklist_df": pd.DataFrame(PREDATORY_JOURNAL_CHECKLIST),
    }


def write_canonical_manuscript(
    llm: BaseChatModel,
    paper_title: str,
    research_question: str,
    field: str,
    citation_style: str = "APA",
    target_journal: str = "Peer-Reviewed International Journal",
    author_guidelines: str = "Standard IMRaD structure; 250-word structured abstract; 5 bullet highlights.",
    verified_references: Optional[List[Dict[str, Any]]] = None,
    literature_context: Optional[str] = None,
    experiment_design_context: Optional[str] = None,
    executed_results_stdout: Optional[str] = None,
    is_demo_data: bool = False,
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a complete, canonically written academic paper with in-text citations
    and a full Reference List built ONLY from `verified_references` and `executed_results_stdout`.
    Also generates Cover Letter, Highlights, and Keywords.
    """
    refs = verified_references or []
    ref_block = build_reference_context_block(refs, style=citation_style, max_refs=30)
    canonical_bib = format_full_bibliography(refs, style=citation_style)

    if executed_results_stdout and executed_results_stdout.strip():
        results_instruction = (
            f"ACTUAL EXECUTED STATISTICAL RESULTS (`stdout` from Experiment Agent; "
            f"Data Status = {'DEMO DATA – not real' if is_demo_data else 'User Empirical Data'}):\n"
            f"```text\n{executed_results_stdout[:5500]}\n```\n"
            "CRITICAL: Report ONLY the exact numbers, means, standard deviations, test statistics, and p-values "
            "from the executed output above. Never invent additional numbers."
        )
    else:
        results_instruction = (
            "NO EMPIRICAL CODE EXECUTION OUTPUT WAS PROVIDED YET.\n"
            "CRITICAL: Do NOT fabricate any sample statistics, p-values, accuracy numbers, or tables. "
            "In Section 4 (Results), provide a clearly labeled **Pre-Registered Analytical & Reporting Template** "
            "with `[INSERT EXECUTED STATISTIC HERE]` placeholders showing exactly where outputs from the Experiment Agent will be inserted."
        )

    system_prompt = (
        "You are the **AI Academic Writing & Publication Agent** inside ResearchAI ('From data to publication.').\n"
        "You write canonical, publication-ready scholarly manuscripts in clear, rigorous academic prose.\n\n"
        "NON-NEGOTIABLE INTEGRITY RULES:\n"
        "1. CITE ONLY FROM THE VERIFIED REFERENCE LIBRARY PROVIDED. Use exact in-text citation keys "
        f"matching **{citation_style}** style (e.g., `(Author et al., Year)` for APA/Harvard or `[1]`, `[2]` for IEEE/Vancouver).\n"
        "2. NEVER fabricate any citation, author, year, or DOI outside the provided library.\n"
        "3. NEVER fabricate empirical data, statistics, p-values, or figures. Use ONLY the executed results provided.\n"
        "4. Write complete, well-developed academic paragraphs for every section so the user can download a complete, canonical paper."
    )

    lit_snip = f"\nLiterature Review Context:\n{literature_context[:2000]}\n" if literature_context else ""
    exp_snip = f"\nExperimental Protocol Context:\n{experiment_design_context[:1800]}\n" if experiment_design_context else ""

    user_prompt = f"""Write a **Complete, Canonical Academic Manuscript** ready for export to Word (DOCX), PDF, and Markdown:

- **Manuscript Title**: {paper_title}
- **Core Research Question**: {research_question}
- **Field / Discipline**: {field}
- **Target Journal**: {target_journal}
- **Citation Style**: {citation_style}
- **Journal Author Guidelines to Follow**: {author_guidelines}
{lit_snip}
{exp_snip}

{results_instruction}

{ref_block}

Please write the complete paper in Markdown with all of the following canonical sections:

# {paper_title}

## Structured Abstract
- **Background**, **Objective**, **Methods**, **Results**, and **Conclusions** (approx. 200–250 words).

## Keywords & Highlights
- **Keywords**: 5–7 indexing keywords separated by semicolons.
- **Research Highlights**: 5 concise bullet points (max 85 characters each) capturing core contributions.

## 1. Introduction
- Context and global significance of the problem, critical synthesis of prior work (with in-text citations from the Verified Reference Library), explicit statement of the research gap, objectives, and key contributions of this paper.

## 2. Literature Review & Theoretical Framework
- Thematic synthesis of the verified literature, comparing existing methodologies and motivating our approach (citing the verified references throughout).

## 3. Materials and Methods
- Study design, data acquisition/cohort description, experimental variables, preprocessing, statistical/machine-learning formulation, validation protocol, and reproducibility specification.

## 4. Results
- Detailed presentation of empirical findings, descriptive statistics, hypothesis tests, regression/ML metrics, and table/figure callouts strictly adhering to the executed results rule above.

## 5. Discussion
- Interpretation of findings in relation to the verified literature, theoretical and practical implications, and why the results matter for {field}.

## 6. Limitations, Threats to Validity & Future Work
- Honest discussion of sample size, generalizability, assumptions, and concrete next steps.

## 7. Conclusion
- Concise synthesis of key takeaways.

## Declarations & Back Matter
- **CRediT Authorship Contribution Statement**
- **Data & Code Availability Statement**
- **Conflict of Interest Disclosure**
- **Responsible AI-Use Disclosure Statement** (explicitly disclosing use of ResearchAI for literature organization, code drafting, and language editing, verified by the human authors)

## References ({citation_style} Style)
- Include the complete, formatted bibliography of all verified references used in the manuscript (with DOIs/URLs).
"""

    manuscript_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    # Ensure the complete canonical bibliography from the Verified Reference Library is always present
    if refs and "## References" not in manuscript_md:
        manuscript_md += f"\n\n## References ({citation_style} Style)\n\n{canonical_bib}\n"

    # Generate Cover Letter
    cover_prompt = f"""Write a formal, persuasive **Journal Submission Cover Letter** addressed to the Editor-in-Chief of **{target_journal}** for the manuscript titled **"{paper_title}"**.
Include:
1. Why the manuscript fits the scope and readership of {target_journal}
2. Summary of novel contributions and empirical rigor
3. Confirmation that the manuscript is original, not under consideration elsewhere, all authors have approved submission, and AI assistance has been transparently disclosed.
"""
    cover_letter_md = invoke_llm_text(
        llm=llm,
        system_prompt="You are an experienced Principal Investigator writing a journal submission cover letter.",
        user_prompt=cover_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "paper_title": paper_title,
        "citation_style": citation_style,
        "target_journal": target_journal,
        "manuscript_md": manuscript_md,
        "cover_letter_md": cover_letter_md,
        "verified_references": refs,
    }


def simulate_peer_review_and_rebuttal(
    llm: BaseChatModel,
    manuscript_md: str,
    target_journal: str = "Peer-Reviewed Journal",
    author_revision_notes: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Simulate constructive, rigorous peer review (Reviewer 1, Reviewer 2, and Associate Editor
    Decision with Major/Minor comments) plus a Point-by-Point Response-to-Reviewers Letter.
    """
    system_prompt = (
        "You are a constructive Academic Peer Review Panel (Associate Editor + 2 Domain Reviewers) "
        f"for {target_journal}. Provide rigorous, actionable major and minor comments on methodology, "
        "statistical validity, clarity, literature coverage, and limitations."
    )

    review_prompt = f"""Conduct a simulated **Double-Anonymous Peer Review** of the following manuscript draft:

```markdown
{manuscript_md[:9000]}
```

Format your report with:
## 1. Associate Editor's Summary & Decision Recommendation
- Overall assessment and decision tier (e.g., *Major Revisions / Minor Revisions Required*)

## 2. Reviewer #1 (Domain & Theoretical Specialist)
- **Summary of Contribution**
- **Major Comments (1–3)**
- **Minor & Editorial Comments (1–4)**

## 3. Reviewer #2 (Methodology, Statistics & Reproducibility Specialist)
- **Summary of Technical Rigor**
- **Major Comments (1–3)**
- **Minor & Reporting Comments (1–4)**
"""
    peer_review_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=review_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    rebuttal_prompt = f"""Based on the simulated Peer Review Report below and the author's revision notes ({author_revision_notes or 'Address all comments thoroughly'}), draft a formal **Point-by-Point Response to Reviewers Letter** showing:
- Each Editor / Reviewer comment quoted
- **Author Response**: Polite, evidence-based explanation
- **Action Taken in Revised Manuscript**: Exact text/section changes to make.

Peer Review Report:
{peer_review_md[:5000]}
"""
    rebuttal_md = invoke_llm_text(
        llm=llm,
        system_prompt="You are a senior academic mentor helping an author draft a respectful, rigorous Point-by-Point Response to Reviewers.",
        user_prompt=rebuttal_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "peer_review_md": peer_review_md,
        "rebuttal_md": rebuttal_md,
    }
