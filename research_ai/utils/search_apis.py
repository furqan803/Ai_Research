"""
utils/search_apis.py
====================
Real academic & bibliographic search clients for ResearchAI:
- arXiv API (via `arxiv` library + direct Atom XML fallback)
- Semantic Scholar Graph API v1
- OpenAlex Works & Sources API (papers + journal recommendations)
- Crossref Works & Journals API (papers + journal recommendations)
- PubMed / NCBI E-Utilities (esearch + esummary + efetch XML)
- Uploaded PDF text & metadata reader (`pypdf`)

CRITICAL RULE:
Never fabricate citations, DOIs, authors, or journal impact factors. Every paper
returned by this module comes directly from a live scholarly API or user-uploaded PDF.
"""

from __future__ import annotations

import io
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import requests
from pypdf import PdfReader

USER_AGENT = "ResearchAI/1.0 (https://github.com/research-ai; mailto:research@research-ai.org)"
DEFAULT_TIMEOUT = 14


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    # Strip HTML/JATS tags often returned in Crossref abstracts
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    d = str(doi).strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    return d.strip().lower()


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct plain-text abstract from OpenAlex `abstract_inverted_index`."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        positions: List[Tuple[int, str]] = []
        for word, idx_list in inverted_index.items():
            for pos in idx_list:
                positions.append((int(pos), word))
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 1. arXiv Search
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Query arXiv API for preprints and papers."""
    if not query.strip():
        return []
    papers: List[Dict[str, Any]] = []
    try:
        import arxiv

        client = arxiv.Client(page_size=max_results, delay_seconds=0.5, num_retries=1)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            doi = _normalize_doi(result.doi)
            year = result.published.year if result.published else None
            authors = [a.name for a in (result.authors or [])][:8]
            url = result.entry_id or (f"https://doi.org/{doi}" if doi else "")
            papers.append(
                {
                    "title": _clean_text(result.title),
                    "authors": authors,
                    "year": year,
                    "venue": f"arXiv ({result.primary_category})" if getattr(result, "primary_category", None) else "arXiv",
                    "doi": doi,
                    "url": url,
                    "pdf_url": getattr(result, "pdf_url", "") or "",
                    "abstract": _clean_text(result.summary),
                    "citation_count": None,
                    "source_db": "arXiv",
                    "is_open_access": True,
                }
            )
    except Exception:
        # Fallback to direct arXiv Atom XML endpoint via requests
        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
                headers={"User-Agent": USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
                    summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
                    published = entry.findtext("atom:published", default="", namespaces=ns)
                    year = int(published[:4]) if published and len(published) >= 4 and published[:4].isdigit() else None
                    link = entry.findtext("atom:id", default="", namespaces=ns)
                    authors = [
                        _clean_text(a.findtext("atom:name", default="", namespaces=ns))
                        for a in entry.findall("atom:author", ns)
                    ]
                    if title:
                        papers.append(
                            {
                                "title": title,
                                "authors": [a for a in authors if a][:8],
                                "year": year,
                                "venue": "arXiv",
                                "doi": "",
                                "url": link,
                                "pdf_url": link.replace("/abs/", "/pdf/") if "/abs/" in link else "",
                                "abstract": summary,
                                "citation_count": None,
                                "source_db": "arXiv",
                                "is_open_access": True,
                            }
                        )
        except Exception:
            pass
    return papers


# ---------------------------------------------------------------------------
# 2. OpenAlex Search
# ---------------------------------------------------------------------------

def search_openalex(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Query OpenAlex Works API for peer-reviewed articles, citations, and OA links."""
    if not query.strip():
        return []
    papers: List[Dict[str, Any]] = []
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": max_results,
                "sort": "relevance_score:desc",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        for item in data.get("results", []):
            title = _clean_text(item.get("display_name") or item.get("title"))
            if not title:
                continue
            doi = _normalize_doi(item.get("doi"))
            year = item.get("publication_year")
            authors = []
            for auth in (item.get("authorships") or [])[:8]:
                name = (auth.get("author") or {}).get("display_name")
                if name:
                    authors.append(name)

            primary_loc = item.get("primary_location") or {}
            source_info = primary_loc.get("source") or {}
            venue = source_info.get("display_name") or "OpenAlex Indexed Venue"
            oa_info = item.get("open_access") or {}
            is_oa = bool(oa_info.get("is_oa"))
            oa_url = oa_info.get("oa_url") or ""

            url = (
                (f"https://doi.org/{doi}" if doi else "")
                or primary_loc.get("landing_page_url")
                or oa_url
                or item.get("id")
                or ""
            )
            abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": doi,
                    "url": url,
                    "pdf_url": oa_url or "",
                    "abstract": _clean_text(abstract),
                    "citation_count": item.get("cited_by_count"),
                    "source_db": "OpenAlex",
                    "is_open_access": is_oa,
                }
            )
    except Exception:
        pass
    return papers


# ---------------------------------------------------------------------------
# 3. Semantic Scholar Search
# ---------------------------------------------------------------------------

def search_semantic_scholar(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Query Semantic Scholar Graph API v1."""
    if not query.strip():
        return []
    papers: List[Dict[str, Any]] = []
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,venue,externalIds,url,abstract,citationCount,isOpenAccess,openAccessPdf",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        for item in data.get("data", []):
            title = _clean_text(item.get("title"))
            if not title:
                continue
            ext_ids = item.get("externalIds") or {}
            doi = _normalize_doi(ext_ids.get("DOI"))
            authors = [a.get("name") for a in (item.get("authors") or []) if a.get("name")][:8]
            oa_pdf = (item.get("openAccessPdf") or {}).get("url") or ""
            url = (f"https://doi.org/{doi}" if doi else "") or item.get("url") or oa_pdf or ""
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": item.get("year"),
                    "venue": item.get("venue") or "Semantic Scholar",
                    "doi": doi,
                    "url": url,
                    "pdf_url": oa_pdf,
                    "abstract": _clean_text(item.get("abstract")),
                    "citation_count": item.get("citationCount"),
                    "source_db": "Semantic Scholar",
                    "is_open_access": bool(item.get("isOpenAccess")),
                }
            )
    except Exception:
        pass
    return papers


# ---------------------------------------------------------------------------
# 4. Crossref Search
# ---------------------------------------------------------------------------

def search_crossref(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Query Crossref Works API for DOI-verified journal articles."""
    if not query.strip():
        return []
    papers: List[Dict[str, Any]] = []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": query,
                "rows": max_results,
                "select": "DOI,title,author,issued,container-title,URL,abstract,is-referenced-by-count",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        items = (resp.json().get("message") or {}).get("items") or []
        for item in items:
            titles = item.get("title") or []
            title = _clean_text(titles[0] if titles else "")
            if not title:
                continue
            doi = _normalize_doi(item.get("DOI"))
            authors = []
            for a in (item.get("author") or [])[:8]:
                given = a.get("given", "")
                family = a.get("family", "")
                full = f"{given} {family}".strip() or a.get("name", "")
                if full:
                    authors.append(full)
            issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
            year = issued[0] if issued and isinstance(issued[0], int) else None
            containers = item.get("container-title") or []
            venue = _clean_text(containers[0]) if containers else "Crossref DOI Registry"
            url = f"https://doi.org/{doi}" if doi else (item.get("URL") or "")
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": doi,
                    "url": url,
                    "pdf_url": "",
                    "abstract": _clean_text(item.get("abstract")),
                    "citation_count": item.get("is-referenced-by-count"),
                    "source_db": "Crossref",
                    "is_open_access": False,
                }
            )
    except Exception:
        pass
    return papers


# ---------------------------------------------------------------------------
# 5. PubMed (NCBI E-utilities) Search
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Query NCBI PubMed E-utilities (esearch + efetch XML) for biomedical & life-science literature."""
    if not query.strip():
        return []
    papers: List[Dict[str, Any]] = []
    try:
        esearch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"},
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if esearch.status_code != 200:
            return []
        id_list = ((esearch.json().get("esearchresult") or {}).get("idlist")) or []
        if not id_list:
            return []

        efetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if efetch.status_code != 200:
            return []

        root = ET.fromstring(efetch.text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID", default="").strip()
            title = _clean_text(article.findtext(".//ArticleTitle", default=""))
            if not title:
                continue

            abstract_parts = [
                _clean_text(ab.text) for ab in article.findall(".//Abstract/AbstractText") if ab.text
            ]
            abstract = " ".join(abstract_parts)

            journal = _clean_text(article.findtext(".//Journal/Title", default="PubMed Indexed Journal"))
            year_str = (
                article.findtext(".//Journal/JournalIssue/PubDate/Year")
                or article.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", default="")[:4]
            )
            year = int(year_str) if year_str and year_str.isdigit() else None

            authors = []
            for au in article.findall(".//AuthorList/Author")[:8]:
                fore = au.findtext("ForeName", default="").strip()
                last = au.findtext("LastName", default="").strip()
                full = f"{fore} {last}".strip()
                if full:
                    authors.append(full)

            doi = ""
            for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if aid.attrib.get("IdType") == "doi" and aid.text:
                    doi = _normalize_doi(aid.text)
                    break

            url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": journal,
                    "doi": doi,
                    "url": url,
                    "pdf_url": "",
                    "abstract": abstract,
                    "citation_count": None,
                    "source_db": "PubMed",
                    "is_open_access": False,
                }
            )
    except Exception:
        pass
    return papers


# ---------------------------------------------------------------------------
# Unified Multi-Database Search + Deduplication + PRISMA Metrics
# ---------------------------------------------------------------------------

def _title_fingerprint(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:85]


def search_all_databases(
    query: str,
    databases: Optional[List[str]] = None,
    max_per_db: int = 6,
    year_from: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search across selected scholarly databases, deduplicate by DOI and normalized title,
    and return both the deduplicated papers and PRISMA identification/deduplication counts.
    """
    selected = databases or ["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"]
    raw_papers: List[Dict[str, Any]] = []
    per_db_counts: Dict[str, int] = {}

    db_dispatch = {
        "OpenAlex": search_openalex,
        "Crossref": search_crossref,
        "arXiv": search_arxiv,
        "Semantic Scholar": search_semantic_scholar,
        "PubMed": search_pubmed,
    }

    for db_name in selected:
        fn = db_dispatch.get(db_name)
        if not fn:
            continue
        results = fn(query, max_results=max_per_db)
        per_db_counts[db_name] = len(results)
        raw_papers.extend(results)

    total_identified = len(raw_papers)

    # Deduplicate by DOI first, then by normalized title fingerprint
    seen_dois = set()
    seen_titles = set()
    deduped: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for p in raw_papers:
        if year_from and p.get("year") and isinstance(p["year"], int) and p["year"] < year_from:
            continue
        doi = _normalize_doi(p.get("doi"))
        tfp = _title_fingerprint(p.get("title", ""))
        if not tfp:
            continue

        if (doi and doi in seen_dois) or (tfp in seen_titles):
            duplicates_removed += 1
            continue

        if doi:
            seen_dois.add(doi)
        seen_titles.add(tfp)
        p["doi"] = doi
        deduped.append(p)

    return {
        "query": query,
        "papers": deduped,
        "per_db_counts": per_db_counts,
        "total_identified": total_identified,
        "duplicates_removed": duplicates_removed,
        "unique_count": len(deduped),
    }


# ---------------------------------------------------------------------------
# Journal Recommendation Search (OpenAlex Sources + Crossref Journals)
# ---------------------------------------------------------------------------

def search_journals(topic_or_field: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Query OpenAlex Sources and Crossref Journals for real scholarly journals matching
    the researcher's topic/field.
    NOTE: Never fabricates Impact Factors or acceptance rates.
    """
    journals: List[Dict[str, Any]] = []
    seen_names = set()

    # 1. OpenAlex Sources
    try:
        resp = requests.get(
            "https://api.openalex.org/sources",
            params={
                "search": topic_or_field,
                "filter": "type:journal",
                "per-page": max_results,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            for src in resp.json().get("results", []):
                name = _clean_text(src.get("display_name"))
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                issn_l = src.get("issn_l") or ""
                publisher = src.get("host_organization_name") or "Academic Publisher"
                is_oa = bool(src.get("is_oa"))
                in_doaj = bool(src.get("is_in_doaj"))
                homepage = src.get("homepage_url") or (
                    f"https://portal.issn.org/resource/ISSN/{issn_l}" if issn_l else "https://openalex.org"
                )
                summary_stats = src.get("summary_stats") or {}
                two_yr_citedness = summary_stats.get("2yr_mean_citedness")
                h_index = summary_stats.get("h_index")

                journals.append(
                    {
                        "name": name,
                        "publisher": publisher,
                        "issn": issn_l,
                        "is_open_access": is_oa,
                        "in_doaj": in_doaj,
                        "homepage_url": homepage,
                        "works_count": src.get("works_count"),
                        "cited_by_count": src.get("cited_by_count"),
                        "two_yr_mean_citedness": round(float(two_yr_citedness), 2) if two_yr_citedness else None,
                        "h_index": h_index,
                        "source_api": "OpenAlex Sources",
                        "metrics_note": "Verify current Journal Impact Factor (Clarivate JCR), CiteScore, and acceptance rate directly on the journal website.",
                    }
                )
    except Exception:
        pass

    # 2. Crossref Journals fallback / supplement
    if len(journals) < max_results:
        try:
            resp = requests.get(
                "https://api.crossref.org/journals",
                params={"query": topic_or_field, "rows": max_results - len(journals)},
                headers={"User-Agent": USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                items = (resp.json().get("message") or {}).get("items") or []
                for item in items:
                    name = _clean_text(item.get("title"))
                    if not name or name.lower() in seen_names:
                        continue
                    seen_names.add(name.lower())
                    issns = item.get("ISSN") or []
                    issn = issns[0] if issns else ""
                    journals.append(
                        {
                            "name": name,
                            "publisher": item.get("publisher") or "Crossref Listed Publisher",
                            "issn": issn,
                            "is_open_access": False,
                            "in_doaj": False,
                            "homepage_url": f"https://portal.issn.org/resource/ISSN/{issn}" if issn else "https://search.crossref.org",
                            "works_count": (item.get("counts") or {}).get("total-dois"),
                            "cited_by_count": None,
                            "two_yr_mean_citedness": None,
                            "h_index": None,
                            "source_api": "Crossref Journals",
                            "metrics_note": "Verify current Journal Impact Factor, APCs, and author guidelines on the official journal website.",
                        }
                    )
        except Exception:
            pass

    return journals[:max_results]


# ---------------------------------------------------------------------------
# PDF Reader for Literature Review Agent (via `pypdf`)
# ---------------------------------------------------------------------------

def extract_pdf_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract text, page count, DOI (if present in text/metadata), and title from an
    uploaded PDF file using `pypdf`.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        meta = reader.metadata or {}
        meta_title = _clean_text(getattr(meta, "title", None) or meta.get("/Title", ""))
        meta_author = _clean_text(getattr(meta, "author", None) or meta.get("/Author", ""))

        pages_text: List[str] = []
        for idx, page in enumerate(reader.pages[:25]):  # Read up to 25 pages for fast extraction
            try:
                t = page.extract_text() or ""
                if t.strip():
                    pages_text.append(f"[Page {idx + 1}]\n{t.strip()}")
            except Exception:
                continue

        full_text = "\n\n".join(pages_text)
        # Look for a DOI pattern in first 2 pages
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", full_text[:6000])
        doi = _normalize_doi(doi_match.group(0).rstrip(".,;)")) if doi_match else ""

        inferred_title = meta_title if (meta_title and len(meta_title) > 5) else filename.rsplit(".", 1)[0]
        year_match = re.search(r"\b(19\d\d|20[0-2]\d)\b", full_text[:3000])
        year = int(year_match.group(1)) if year_match else None

        return {
            "filename": filename,
            "title": inferred_title,
            "authors": [meta_author] if meta_author else ["Uploaded PDF Author"],
            "year": year,
            "venue": "Uploaded PDF Document",
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else f"uploaded://{filename}",
            "num_pages": num_pages,
            "text_excerpt": full_text[:15000],
            "abstract": _clean_text(full_text[:1200]),
            "source_db": "Uploaded PDF",
            "is_open_access": True,
        }
    except Exception as e:
        return {
            "filename": filename,
            "title": filename,
            "authors": [],
            "year": None,
            "venue": "Uploaded PDF (Read Error)",
            "doi": "",
            "url": f"uploaded://{filename}",
            "num_pages": 0,
            "text_excerpt": "",
            "abstract": f"Could not parse PDF: {e}",
            "source_db": "Uploaded PDF",
            "is_open_access": False,
        }
