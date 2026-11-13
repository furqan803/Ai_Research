"""
utils/reference_library.py
==========================
Verified Reference Library & Canonical Citation Formatter for ResearchAI.

CRITICAL INTEGRITY RULE:
Only papers returned by real scholarly search APIs (arXiv, OpenAlex, Semantic Scholar,
Crossref, PubMed) or explicitly uploaded PDFs with extracted metadata may enter the
Verified Reference Library. Subsequent agents (Literature Review, Academic Writing)
are strictly bound to cite ONLY from this verified library.

Supported Citation Styles:
- APA (7th Edition)
- IEEE (Numbered `[1]`, `[2]`, ...)
- Harvard
- MLA (9th Edition)
- Vancouver (Biomedical numbered)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _format_authors_apa(authors: List[str]) -> str:
    if not authors:
        return "Anonymous"
    formatted = []
    for name in authors[:7]:
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join(f"{p[0].upper()}." for p in parts[:-1] if p)
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(name.strip())
    if len(authors) > 7:
        return ", ".join(formatted[:6]) + ", ... " + formatted[-1]
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + ", & " + formatted[-1]


def _format_authors_ieee(authors: List[str]) -> str:
    if not authors:
        return "Anonymous"
    formatted = []
    for name in authors[:6]:
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join(f"{p[0].upper()}." for p in parts[:-1] if p)
            formatted.append(f"{initials} {last}")
        else:
            formatted.append(name.strip())
    if len(authors) > 6:
        return ", ".join(formatted[:3]) + " et al."
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def _format_authors_vancouver(authors: List[str]) -> str:
    if not authors:
        return "Anonymous"
    formatted = []
    for name in authors[:6]:
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0].upper() for p in parts[:-1] if p)
            formatted.append(f"{last} {initials}")
        else:
            formatted.append(name.strip())
    if len(authors) > 6:
        return ", ".join(formatted) + ", et al"
    return ", ".join(formatted)


def make_in_text_key(paper: Dict[str, Any], index: int, style: str = "APA") -> str:
    """Return canonical in-text citation key (e.g., `(Smith et al., 2023)` or `[1]`)."""
    s = (style or "APA").upper()
    if s in ("IEEE", "VANCOUVER"):
        return f"[{index}]"

    authors = paper.get("authors") or []
    year = paper.get("year") or "n.d."
    if not authors:
        short_title = (paper.get("title") or "Study")[:24].strip()
        return f"({short_title}, {year})"

    first_last = authors[0].strip().split()[-1]
    if len(authors) == 1:
        if s == "MLA":
            return f"({first_last})"
        return f"({first_last}, {year})"
    elif len(authors) == 2:
        second_last = authors[1].strip().split()[-1]
        if s == "MLA":
            return f"({first_last} and {second_last})"
        return f"({first_last} & {second_last}, {year})"
    else:
        if s == "MLA":
            return f"({first_last} et al.)"
        return f"({first_last} et al., {year})"


def format_citation(paper: Dict[str, Any], index: int = 1, style: str = "APA") -> str:
    """
    Format a single verified paper dictionary into a canonical bibliographic entry.
    Includes DOI or verified source URL.
    """
    s = (style or "APA").upper()
    title = (paper.get("title") or "Untitled Work").strip().rstrip(".")
    authors = paper.get("authors") or []
    year = paper.get("year") or "n.d."
    venue = (paper.get("venue") or paper.get("source_db") or "Scholarly Venue").strip()
    doi = (paper.get("doi") or "").strip()
    url = (paper.get("url") or "").strip()
    link_part = f"https://doi.org/{doi}" if doi else url

    if s == "IEEE":
        au_str = _format_authors_ieee(authors)
        base = f"[{index}] {au_str}, \"{title},\" *{venue}*, {year}."
        if link_part:
            base += f" Available: {link_part}"
        return base

    elif s == "HARVARD":
        au_str = _format_authors_apa(authors).replace("&", "and")
        base = f"{au_str} ({year}) '{title}', *{venue}*."
        if link_part:
            base += f" Available at: {link_part}"
        return base

    elif s == "MLA":
        au_str = _format_authors_apa(authors)
        base = f"{au_str}. \"{title}.\" *{venue}*, {year}."
        if link_part:
            base += f" {link_part}."
        return base

    elif s == "VANCOUVER":
        au_str = _format_authors_vancouver(authors)
        base = f"{index}. {au_str}. {title}. {venue}. {year}."
        if doi:
            base += f" doi:{doi}"
        elif link_part:
            base += f" Available from: {link_part}"
        return base

    else:  # Default APA 7th
        au_str = _format_authors_apa(authors)
        base = f"{au_str} ({year}). {title}. *{venue}*."
        if link_part:
            base += f" {link_part}"
        return base


def merge_into_library(
    existing_library: List[Dict[str, Any]],
    new_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge newly retrieved papers into the session's verified reference library,
    avoiding duplicates by DOI or normalized title and assigning stable `ref_id`s.
    """
    library = list(existing_library or [])
    seen_dois = {
        (p.get("doi") or "").strip().lower()
        for p in library
        if (p.get("doi") or "").strip()
    }
    seen_titles = {
        re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())[:85]
        for p in library
        if p.get("title")
    }

    for p in new_papers or []:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        doi = (p.get("doi") or "").strip().lower()
        tfp = re.sub(r"[^a-z0-9]", "", title.lower())[:85]
        if (doi and doi in seen_dois) or (tfp and tfp in seen_titles):
            continue
        if doi:
            seen_dois.add(doi)
        if tfp:
            seen_titles.add(tfp)
        entry = dict(p)
        entry["ref_id"] = f"REF-{len(library) + 1}"
        library.append(entry)

    # Ensure sequential ref_id numbering
    for idx, item in enumerate(library, start=1):
        item["ref_id"] = f"REF-{idx}"
        item["ref_number"] = idx

    return library


def build_reference_context_block(
    library: List[Dict[str, Any]],
    style: str = "APA",
    max_refs: int = 25,
) -> str:
    """
    Create a structured prompt block of verified references for LLM agents so they can
    cite ONLY verified papers using exact in-text keys and produce a canonical Reference List.
    """
    if not library:
        return (
            "VERIFIED REFERENCE LIBRARY IS CURRENTLY EMPTY.\n"
            "CRITICAL RULE: Do NOT invent or fabricate any citations, DOIs, or author names. "
            "State explicitly where empirical citations should be inserted once literature search is run."
        )

    lines = [
        f"VERIFIED REFERENCE LIBRARY ({min(len(library), max_refs)} real papers from scholarly APIs; Citation Style = {style}):",
        "CRITICAL RULE: You MUST cite ONLY from the papers listed below. Never invent any external author, year, or DOI.",
        "",
    ]
    for idx, p in enumerate(library[:max_refs], start=1):
        in_text = make_in_text_key(p, idx, style=style)
        full_ref = format_citation(p, idx, style=style)
        abstract_snip = (p.get("abstract") or "No abstract provided in metadata.")[:320]
        lines.append(
            f"- [{p.get('ref_id', f'REF-{idx}')}] In-text key: `{in_text}` | Full citation: {full_ref}\n"
            f"  Source DB: {p.get('source_db', 'API')} | Summary: {abstract_snip}"
        )
    return "\n".join(lines)


def format_full_bibliography(library: List[Dict[str, Any]], style: str = "APA") -> str:
    """Return a clean Markdown bibliography list for all papers in the verified library."""
    if not library:
        return "_No verified references in the library yet. Run the Discovery or Literature Review Agent to populate real DOI/API-backed references._"
    entries = []
    for idx, p in enumerate(library, start=1):
        entries.append(f"{idx}. {format_citation(p, index=idx, style=style)}")
    return "\n".join(entries)
