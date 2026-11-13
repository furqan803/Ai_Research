"""
app.py
======
ResearchAI — Five AI Agents. One Smarter Research Journey.
"From idea to impact — your AI research team."
Footer Motto: "Higher Research, Brighter Tomorrow."

Main Streamlit Multi-Page Application:
- Home (5 Colour-Coded Agent Cards, 6-Step Workflow, Why It Matters, Who Can Use It, Impact Areas, Bigger Vision)
- 1. AI Research Discovery Agent ("Find what matters. Discover new possibilities.")
- 2. AI Literature Review Agent ("Turn thousands of papers into clear insights.")
- 3. AI Research Experiment Agent ("Design. Analyze. Discover." + Safe Sandboxed Code Runner + Biopython)
- 4. AI Academic Writing & Publication Agent ("From data to publication." + Canonical Paper + Verified References + Peer Review + Journal Fit)
- 5. AI PhD Supervisor Agent ("Your always-available research mentor." + 5-Pillar Roadmap + Defense Kit + Ethics + 24/7 Mentor Chat)
- Full Workflow & Impact (LangGraph 6-Step Orchestrator with Human-Approval Checkpoints + Step 6 Create Impact)
- Project Tracker (Kanban / Status Table from Idea -> Published with Downloadable Summary)
- Export Center (One-Click Canonical Paper & Complete Dossier Export in DOCX, PDF, Markdown, and .py)
- User Guide & Responsible AI Ethics (Full interactive documentation & provider comparison table)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.discovery import run_discovery_agent
from agents.experiment import (
    generate_custom_analysis_code,
    generate_experimental_design,
    interpret_executed_results,
    run_bioinformatics_sequence_stats,
)
from agents.literature import run_literature_agent
from agents.supervisor import (
    RESEARCH_ETHICS_MODULES,
    generate_defense_and_presentation_kit,
    generate_phd_roadmap_and_critique,
    stream_supervisor_chat,
)
from agents.writing_publication import (
    PREDATORY_JOURNAL_CHECKLIST,
    recommend_journals_with_fit,
    simulate_peer_review_and_rebuttal,
    write_canonical_manuscript,
)
from graph.workflow import (
    ResearchWorkflowState,
    build_research_workflow_graph,
    generate_impact_package,
    run_single_workflow_stage,
)
from utils.code_runner import (
    build_default_analysis_script,
    execute_sandboxed_python,
    validate_code_safety,
)
from utils.reference_library import format_full_bibliography, merge_into_library
from utils.ui import (
    get_active_llm,
    init_session_state,
    load_custom_css,
    render_academic_disclaimer,
    render_download_toolbar,
    render_footer,
    render_hero_header,
    render_sidebar,
    render_workflow_stepper,
)

# ---------------------------------------------------------------------------
# Page Config & Initialization
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchAI — Five AI Agents. One Smarter Research Journey.",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
load_custom_css()

active_page = render_sidebar()
render_hero_header()
render_workflow_stepper()
render_academic_disclaimer()


# ===========================================================================
# PAGE 0: HOME
# ===========================================================================

def page_home() -> None:
    st.markdown("## 🌟 Meet Your Five Specialized AI Research Agents")
    st.caption(
        "Designed for BS students, MS/PhD candidates, professors, universities, research institutes, "
        "industry R&D departments, and the global research community."
    )

    # Row 1: 3 Agent Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-discovery">
              <h3>🔵 1. AI Research Discovery Agent</h3>
              <div class="rai-agent-tagline">“Find what matters. Discover new possibilities.”</div>
              <ul>
                <li>Scans arXiv, Semantic Scholar, OpenAlex, Crossref &amp; PubMed in real time</li>
                <li>Maps publication counts per year, key authors &amp; top venues as charts</li>
                <li>Pinpoints evidence-backed research gaps &amp; contradictions</li>
                <li>Proposes 5 novel research ideas with RQs, novelty checks &amp; feasibility</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “5 novel PhD topic ideas in AI-based drought prediction for wheat”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Discovery Agent →", key="home_btn_disc", use_container_width=True):
            st.session_state["active_page"] = "1. Discovery Agent"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-literature">
              <h3>🟢 2. AI Literature Review Agent</h3>
              <div class="rai-agent-tagline">“Turn thousands of papers into clear insights.”</div>
              <ul>
                <li>PRISMA-style flow (Identified &rarr; Deduplicated &rarr; Screened &rarr; Included)</li>
                <li>Screens papers with explicit inclusion/exclusion reasons per study</li>
                <li>Extracts structured tables (aim, methods, dataset, results, limitations) + reads PDFs</li>
                <li>Builds a Verified Reference Library with real DOIs for downstream agents</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “A 40-page systematic review draft with tables and key findings”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Literature Agent →", key="home_btn_lit", use_container_width=True):
            st.session_state["active_page"] = "2. Literature Agent"
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-experiment">
              <h3>🟠 3. AI Research Experiment Agent</h3>
              <div class="rai-agent-tagline">“Design. Analyze. Discover.”</div>
              <ul>
                <li>Designs hypotheses ($H_0/H_1$), variables, controls, sample size &amp; protocols</li>
                <li>Runs real statistical &amp; ML analysis on CSV/Excel + Biopython sequences</li>
                <li>Safe subprocess runner: preview code first, click Run, zero fabricated numbers</li>
                <li>Exports publication figures, tables, methods draft &amp; reproducible <code>.py</code> code</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “Complete analysis pipeline with figures, tables and methods section”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Experiment Agent →", key="home_btn_exp", use_container_width=True):
            st.session_state["active_page"] = "3. Experiment Agent"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: 2 Agent Cards + Full Workflow Card
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-writing">
              <h3>🟣 4. AI Academic Writing &amp; Publication Agent</h3>
              <div class="rai-agent-tagline">“From data to publication.”</div>
              <ul>
                <li>Writes canonical manuscripts (Abstract to References) citing verified DOIs</li>
                <li>Recommends journals via Crossref/OpenAlex with scope-fit &amp; OA checks</li>
                <li>Simulates 2-reviewer + editor peer review &amp; writes response-to-reviewers letters</li>
                <li>Formats in APA, IEEE, Harvard, MLA, Vancouver + exports DOCX / PDF / MD</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “A publication-ready manuscript with journal recommendations”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Writing Agent →", key="home_btn_wrt", use_container_width=True):
            st.session_state["active_page"] = "4. Writing & Publication"
            st.rerun()

    with c5:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-supervisor">
              <h3>🩵 5. AI PhD Supervisor Agent</h3>
              <div class="rai-agent-tagline">“Your always-available research mentor.”</div>
              <ul>
                <li>Builds a personalized 5-pillar roadmap (Plan, Guide, Feedback, Improve, Succeed)</li>
                <li>Challenges assumptions with friendly Socratic critical questions</li>
                <li>Prepares slide outlines, poster layouts &amp; mock viva defense Q&amp;As</li>
                <li>Teaches research ethics (CRediT, AI disclosure, integrity) + 24/7 mentor chat</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “A 3-year PhD plan with milestones and continuous feedback”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch PhD Supervisor →", key="home_btn_sup", use_container_width=True):
            st.session_state["active_page"] = "5. PhD Supervisor"
            st.rerun()

    with c6:
        st.markdown(
            """
            <div class="rai-agent-card rai-card-discovery" style="border-top-color:#38bdf8;">
              <h3>🚀 6. Complete LangGraph Pipeline &amp; Impact</h3>
              <div class="rai-agent-tagline">“From raw idea to global scientific &amp; societal impact.”</div>
              <ul>
                <li>Orchestrates all 5 agents sequentially with human-approval checkpoints</li>
                <li>Automatically passes verified citations &amp; executed stats across agents</li>
                <li>Generates Step 6 Impact: Lay summary, LinkedIn post, policy brief &amp; press release</li>
                <li>Includes ORCID / Google Scholar / preprint visibility &amp; citation checklist</li>
              </ul>
              <div class="rai-sample-output">
                <strong>Sample Output:</strong> “End-to-end research dossier + public &amp; policy impact kit”
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Full Workflow →", key="home_btn_wf", use_container_width=True):
            st.session_state["active_page"] = "Full Workflow & Impact"
            st.rerun()

    st.markdown("---")

    # The Complete Research Workflow (6 steps with arrows)
    st.markdown("## 🔄 The Complete Research Workflow")
    st.markdown(
        """
        <div class="rai-info-box" style="text-align:center; font-size:1.03rem; padding:1.25rem;">
          <strong>1. Discover Ideas</strong> &nbsp;&rarr;&nbsp;
          <strong>2. Review Literature</strong> &nbsp;&rarr;&nbsp;
          <strong>3. Design &amp; Analyze</strong> &nbsp;&rarr;&nbsp;
          <strong>4. Write &amp; Publish</strong> &nbsp;&rarr;&nbsp;
          <strong>5. Get Mentored</strong> &nbsp;&rarr;&nbsp;
          <strong>6. Create Impact</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Why It Matters, Who Can Use It, Impact Areas
    col_why, col_who, col_imp = st.columns(3)
    with col_why:
        st.markdown(
            """
            <div class="rai-info-box">
              <h3>💡 Why It Matters</h3>
              <ul>
                <li><strong>Saves months of manual work:</strong> Automates multi-database literature discovery, deduplication, and formatting.</li>
                <li><strong>Improves research quality:</strong> Enforces PRISMA screening, power reasoning, assumption checks, and critical review.</li>
                <li><strong>Increases publication success:</strong> Aligns manuscripts with real journal scopes and pre-tests via simulated peer review.</li>
                <li><strong>Democratizes world-class support:</strong> Gives every student and lab access to a 24/7 AI research team — including with free LLM tiers.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_who:
        st.markdown(
            """
            <div class="rai-info-box">
              <h3>👥 Who Can Use It</h3>
              <ul>
                <li><strong>BS &amp; MS Students:</strong> Learn canonical research methods and write first thesis chapters.</li>
                <li><strong>PhD Students &amp; Candidates:</strong> Discover novel gaps, run systematic reviews, and prepare for viva defense.</li>
                <li><strong>Researchers &amp; Professors:</strong> Accelerate grant proposals, lab supervision, and collaborative papers.</li>
                <li><strong>Universities &amp; Research Institutes:</strong> Standardize research training and ethical AI practices.</li>
                <li><strong>R&amp;D Departments (Industry) &amp; Global Community:</strong> Translate frontier literature into applied innovation.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_imp:
        st.markdown(
            """
            <div class="rai-info-box">
              <h3>🌍 Impact Areas</h3>
              <ul>
                <li><strong>🌾 Agriculture &amp; Food Security:</strong> Crop drought resilience, precision agronomy, soil &amp; yield modeling.</li>
                <li><strong>🧬 Healthcare &amp; Life Sciences:</strong> Clinical biostatistics, genomics, bioinformatics &amp; epidemiology.</li>
                <li><strong>⚙️ Engineering &amp; Technology:</strong> AI/ML systems, robotics, renewable energy &amp; materials science.</li>
                <li><strong>🌱 Climate Change &amp; Environment:</strong> Earth observation, biodiversity, carbon &amp; hydrological modeling.</li>
                <li><strong>🏛️ Social Sciences, Policy &amp; Humanities:</strong> Econometrics, education, public policy &amp; behavioral studies.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rai-hero" style="text-align:center; justify-content:center;">
          <div>
            <h2 style="margin:0; color:#38bdf8;">✨ A Bigger Vision</h2>
            <p style="font-size:1.25rem; font-weight:700; color:#f8fafc; margin:0.4rem 0;">
              “Because better research creates a better world.”
            </p>
            <p style="color:#cbd5e1; margin:0;">
              Higher Research, Brighter Tomorrow.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1: AI RESEARCH DISCOVERY AGENT
# ===========================================================================

def page_discovery() -> None:
    st.markdown("## 🔵 Agent 1: AI Research Discovery Agent")
    st.caption("“Find what matters. Discover new possibilities.”")

    with st.form("discovery_form"):
        c1, c2 = st.columns(2)
        with c1:
            field = st.text_input(
                "Research Field / Domain",
                value=st.session_state.get("researcher_field", "Agricultural & Environmental AI"),
            )
            keywords = st.text_input(
                "Specific Keywords / Topic Focus",
                value="AI-based drought prediction for wheat using multi-spectral remote sensing",
            )
            career_stage = st.selectbox(
                "Career Stage",
                options=[
                    "BS / Undergraduate Student",
                    "MS / Master's Student",
                    "PhD Student / Candidate",
                    "Postdoctoral Researcher",
                    "Professor / Principal Investigator",
                    "Industry R&D Scientist",
                ],
                index=2,
            )
        with c2:
            available_resources = st.text_input(
                "Available Compute, Lab & Data Resources",
                value=st.session_state.get(
                    "available_resources",
                    "University GPU workstation, public Sentinel-2/MODIS satellite data, agronomic trial records",
                ),
            )
            preferred_depth = st.selectbox(
                "Preferred Analysis Depth",
                options=[
                    "Rapid Exploratory Scan (BS/MS Project)",
                    "Deep PhD-Level Exploration (Novelty & Gap Rigor)",
                    "Grant / Flagship Lab Proposal Depth",
                ],
                index=1,
            )
            selected_dbs = st.multiselect(
                "Scholarly Databases to Scan",
                options=["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
                default=["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
            )
            max_per_db = st.slider("Papers per Database", min_value=3, max_value=12, value=5)

        submitted = st.form_submit_button("🔍 Scan Databases, Map Trends & Generate 5 Novel Research Ideas", use_container_width=True)

    if submitted:
        try:
            llm = get_active_llm()
            with st.spinner("Scanning arXiv, OpenAlex, Semantic Scholar, Crossref & PubMed and synthesizing 5 novel research ideas..."):
                result = run_discovery_agent(
                    llm=llm,
                    field=field,
                    keywords=keywords,
                    career_stage=career_stage,
                    available_resources=available_resources,
                    preferred_depth=preferred_depth,
                    databases=selected_dbs,
                    max_per_db=max_per_db,
                    citation_style=st.session_state.get("citation_style", "APA"),
                    existing_library=st.session_state.get("verified_references", []),
                    provider_name=st.session_state.get("llm_provider", ""),
                    base_url=st.session_state.get("llm_base_url"),
                )
                st.session_state["discovery_output"] = result
                st.session_state["verified_references"] = result["verified_references"]
                st.session_state["researcher_field"] = field
                st.success(
                    f"Retrieved {len(result['papers'])} unique verified papers across {len(selected_dbs)} databases "
                    f"(added to Verified Reference Library: {len(result['verified_references'])} total)."
                )
        except Exception as e:
            st.error(str(e))

    disc_out = st.session_state.get("discovery_output")
    if disc_out:
        trends = disc_out["trends"]
        st.markdown("### 📊 Bibliometric Trends from Retrieved Literature")
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            df_yr = trends["years"]
            if not df_yr.empty:
                fig_yr = px.bar(
                    df_yr,
                    x="Year",
                    y="Publications",
                    title="Publications per Year",
                    color_discrete_sequence=["#3b82f6"],
                    template="plotly_dark",
                )
                st.plotly_chart(fig_yr, use_container_width=True)
            else:
                st.info("No publication year metadata available.")
        with ch2:
            df_au = trends["authors"]
            if not df_au.empty:
                fig_au = px.bar(
                    df_au.head(8),
                    x="Paper Count",
                    y="Author",
                    orientation="h",
                    title="Top Contributing Authors",
                    color_discrete_sequence=["#14b8a6"],
                    template="plotly_dark",
                )
                st.plotly_chart(fig_au, use_container_width=True)
            else:
                st.info("No author frequency data available.")
        with ch3:
            df_vn = trends["venues"]
            if not df_vn.empty:
                fig_vn = px.bar(
                    df_vn.head(8),
                    x="Paper Count",
                    y="Venue / Journal",
                    orientation="h",
                    title="Top Venues / Sources",
                    color_discrete_sequence=["#8b5cf6"],
                    template="plotly_dark",
                )
                st.plotly_chart(fig_vn, use_container_width=True)
            else:
                st.info("No venue frequency data available.")

        with st.expander(f"📚 View Retrieved Verified Papers ({len(disc_out['papers'])} unique papers)", expanded=False):
            papers_df = pd.DataFrame(
                [
                    {
                        "Title": p.get("title"),
                        "Authors": ", ".join((p.get("authors") or [])[:4]),
                        "Year": p.get("year"),
                        "Venue": p.get("venue"),
                        "Source DB": p.get("source_db"),
                        "DOI / URL": f"https://doi.org/{p['doi']}" if p.get("doi") else p.get("url"),
                    }
                    for p in disc_out["papers"]
                ]
            )
            st.dataframe(papers_df, use_container_width=True)

        st.markdown("### 🧠 Discovery Report: Trends, Gaps & 5 Novel Research Ideas")
        st.markdown(disc_out["report_md"])

        render_download_toolbar(
            title=f"Research Discovery & 5 Novel Ideas: {disc_out['query']}",
            content_md=disc_out["report_md"],
            file_prefix="research_discovery_report",
            references=disc_out["verified_references"],
        )


# ===========================================================================
# PAGE 2: AI LITERATURE REVIEW AGENT
# ===========================================================================

def page_literature() -> None:
    st.markdown("## 🟢 Agent 2: AI Literature Review Agent")
    st.caption("“Turn thousands of papers into clear insights.”")

    default_query = (
        (st.session_state.get("discovery_output") or {}).get("query")
        or f"{st.session_state.get('researcher_field', '')} drought prediction wheat machine learning"
    ).strip()

    with st.form("literature_form"):
        topic_query = st.text_input("Literature Search Query", value=default_query)
        c1, c2 = st.columns(2)
        with c1:
            inc_criteria = st.text_area(
                "Inclusion Criteria (comma or line separated)",
                value="wheat, drought, machine learning, remote sensing, yield, prediction, empirical model",
                height=85,
            )
            min_year = st.number_input("Minimum Publication Year", min_value=1990, max_value=2026, value=2018)
        with c2:
            exc_criteria = st.text_area(
                "Exclusion Criteria (comma or line separated)",
                value="corrigendum, erratum, book review, unrelated clinical trial",
                height=85,
            )
            max_per_db = st.slider("Max Papers per Database", min_value=3, max_value=12, value=5)

        selected_dbs = st.multiselect(
            "Databases to Search",
            options=["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
            default=["OpenAlex", "Crossref", "arXiv", "Semantic Scholar", "PubMed"],
        )
        uploaded_files = st.file_uploader(
            "Upload Additional Full-Text PDFs (Optional — parsed via `pypdf` and added to Verified Reference Library)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        run_lit = st.form_submit_button("📚 Run PRISMA Screening, Extract Evidence Tables & Write Systematic Review", use_container_width=True)

    if run_lit:
        try:
            llm = get_active_llm()
            pdf_tuples: List[Tuple[bytes, str]] = []
            for uf in uploaded_files or []:
                pdf_tuples.append((uf.read(), uf.name))

            disc_context = (st.session_state.get("discovery_output") or {}).get("report_md")
            with st.spinner("Searching APIs, deduplicating, running PRISMA screening, extracting tables & writing systematic review..."):
                lit_res = run_literature_agent(
                    llm=llm,
                    topic_query=topic_query,
                    inclusion_criteria=inc_criteria,
                    exclusion_criteria=exc_criteria,
                    min_year=int(min_year),
                    databases=selected_dbs,
                    max_per_db=max_per_db,
                    uploaded_pdfs=pdf_tuples,
                    citation_style=st.session_state.get("citation_style", "APA"),
                    existing_library=st.session_state.get("verified_references", []),
                    discovery_context=disc_context,
                    provider_name=st.session_state.get("llm_provider", ""),
                    base_url=st.session_state.get("llm_base_url"),
                )
                st.session_state["literature_output"] = lit_res
                st.session_state["verified_references"] = lit_res["verified_references"]
                st.success(
                    f"PRISMA Flow Complete: {lit_res['prisma_counts']['Identified from APIs & PDFs']} identified -> "
                    f"{lit_res['prisma_counts']['Records Screened']} screened -> "
                    f"{lit_res['prisma_counts']['Studies Included in Review']} included in synthesis."
                )
        except Exception as e:
            st.error(str(e))

    lit_out = st.session_state.get("literature_output")
    if lit_out:
        prisma = lit_out["prisma_counts"]
        st.markdown("### 📐 PRISMA-Style Literature Selection Flow")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1. Identified", prisma["Identified from APIs & PDFs"])
        m2.metric("2. Duplicates Removed", prisma["Duplicates Removed"])
        m3.metric("3. Screened", prisma["Records Screened"])
        m4.metric("4. Excluded", prisma["Records Excluded"])
        m5.metric("5. Included", prisma["Studies Included in Review"])

        fig_prisma = go.Figure(
            go.Funnel(
                y=[
                    "Identified (APIs + PDFs)",
                    "After Deduplication (Screened)",
                    "Included in Final Synthesis",
                ],
                x=[
                    prisma["Identified from APIs & PDFs"],
                    prisma["Records Screened"],
                    prisma["Studies Included in Review"],
                ],
                textinfo="value+percent initial",
                marker={"color": ["#3b82f6", "#14b8a6", "#10b981"]},
            )
        )
        fig_prisma.update_layout(
            title="PRISMA Screening Funnel",
            template="plotly_dark",
            height=310,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_prisma, use_container_width=True)

        tab_ev, tab_scr, tab_ref = st.tabs(
            [
                "📋 Structured Evidence Extraction Table",
                "🔍 Per-Paper Screening Log (Inclusion/Exclusion Decisions)",
                f"✅ Verified Reference Library ({len(lit_out['verified_references'])} Papers)",
            ]
        )
        with tab_ev:
            st.dataframe(lit_out["evidence_df"], use_container_width=True)
        with tab_scr:
            st.dataframe(lit_out["screening_df"], use_container_width=True)
        with tab_ref:
            st.markdown(
                format_full_bibliography(
                    lit_out["verified_references"],
                    style=st.session_state.get("citation_style", "APA"),
                )
            )

        st.markdown("### 📖 Systematic Literature Review Draft")
        st.markdown(lit_out["review_md"])

        render_download_toolbar(
            title=f"Systematic Literature Review: {lit_out['topic_query']}",
            content_md=lit_out["review_md"],
            file_prefix="systematic_literature_review",
            references=lit_out["verified_references"],
        )


# ===========================================================================
# PAGE 3: AI RESEARCH EXPERIMENT AGENT
# ===========================================================================

def page_experiment() -> None:
    st.markdown("## 🟠 Agent 3: AI Research Experiment Agent")
    st.caption("“Design. Analyze. Discover.” — Zero fabricated numbers: every statistic comes from executed Python code.")

    default_rq = (
        (st.session_state.get("literature_output") or {}).get("topic_query")
        or (st.session_state.get("discovery_output") or {}).get("query")
        or "How do drought stress treatments affect soil moisture, canopy NDVI, proline accumulation, and grain yield across wheat varieties?"
    )

    rq = st.text_input("Research Question / Experimental Objective", value=default_rq)
    data_mode = st.radio(
        "Choose Data Mode:",
        options=[
            "Upload CSV / Excel Dataset (Real Empirical Data)",
            "Load Labeled Synthetic Demo Dataset ('DEMO DATA – not real')",
            "Experimental Design Only (No Data Execution Yet)",
        ],
        horizontal=True,
    )

    df_active: pd.DataFrame | None = None
    is_demo = False

    if data_mode == "Upload CSV / Excel Dataset (Real Empirical Data)":
        up_file = st.file_uploader("Upload CSV (.csv) or Excel (.xlsx)", type=["csv", "xlsx", "xls"])
        if up_file is not None:
            try:
                if up_file.name.lower().endswith(".csv"):
                    df_active = pd.read_csv(up_file)
                else:
                    df_active = pd.read_excel(up_file)
                is_demo = False
                st.success(f"Loaded user dataset `{up_file.name}`: {df_active.shape[0]} rows × {df_active.shape[1]} columns.")
                st.dataframe(df_active.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse uploaded file: {e}")

    elif data_mode == "Load Labeled Synthetic Demo Dataset ('DEMO DATA – not real')":
        demo_path = Path(__file__).resolve().parent / "sample_data" / "wheat_drought_trial_demo.csv"
        if demo_path.exists():
            df_active = pd.read_csv(demo_path)
            is_demo = True
            st.warning(
                "⚠️ **DEMO DATA – not real:** Loaded synthetic wheat drought trial dataset (`wheat_drought_trial_demo.csv`, "
                f"{df_active.shape[0]} rows × {df_active.shape[1]} columns) strictly for testing the statistical & ML code pipeline."
            )
            st.dataframe(df_active.head(8), use_container_width=True)

    # Step 3A: Generate Experimental Design Protocol
    c_des, c_code = st.columns(2)
    with c_des:
        if st.button("🧪 1. Generate Experimental Design Protocol", use_container_width=True):
            try:
                llm = get_active_llm()
                ds_summary = (
                    f"{'DEMO DATA – not real' if is_demo else 'User Dataset'}: {df_active.shape[0]} rows, columns={list(df_active.columns)}"
                    if df_active is not None
                    else "Design-Only Mode (no dataset loaded)"
                )
                lit_ctx = (st.session_state.get("literature_output") or {}).get("review_md")
                with st.spinner("Designing hypotheses, variables, controls, power reasoning & validity protocol..."):
                    design_md = generate_experimental_design(
                        llm=llm,
                        research_question=rq,
                        field=st.session_state.get("researcher_field", ""),
                        available_resources=st.session_state.get("available_resources", ""),
                        dataset_summary=ds_summary,
                        literature_context=lit_ctx,
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    cur_exp = st.session_state.get("experiment_output") or {}
                    cur_exp["design_md"] = design_md
                    cur_exp["is_demo"] = is_demo
                    if df_active is not None and not cur_exp.get("code"):
                        cur_exp["code"] = build_default_analysis_script(df_active, is_demo=is_demo)
                    st.session_state["experiment_output"] = cur_exp
            except Exception as e:
                st.error(str(e))

    with c_code:
        if df_active is not None:
            if st.button("🐍 2. Generate / Refresh Python Analysis Script (Preview Before Run)", use_container_width=True):
                try:
                    llm = get_active_llm()
                    with st.spinner("Writing safe, reproducible Python statistical & ML script..."):
                        code_str = generate_custom_analysis_code(
                            llm=llm,
                            df=df_active,
                            analysis_goal=rq,
                            is_demo=is_demo,
                            provider_name=st.session_state.get("llm_provider", ""),
                            base_url=st.session_state.get("llm_base_url"),
                        )
                        cur_exp = st.session_state.get("experiment_output") or {}
                        cur_exp["code"] = code_str
                        cur_exp["is_demo"] = is_demo
                        st.session_state["experiment_output"] = cur_exp
                except Exception as e:
                    st.error(str(e))

    exp_out = st.session_state.get("experiment_output") or {}

    if exp_out.get("design_md"):
        with st.expander("📐 View Experimental Design Protocol (Hypotheses, Power & Validity)", expanded=True):
            st.markdown(exp_out["design_md"])

    # Step 3B: Code Preview + Explicit "Run" Button (Mandatory Safety Requirement)
    if df_active is not None:
        st.markdown("### 🔒 Safe Sandboxed Python Code Runner (Inspect Code Before Running)")
        st.caption(
            "Safety policy: Runs in an isolated temporary directory with a 25s timeout, network sockets disabled, "
            "and an allow-list of scientific libraries (`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `biopython`)."
        )
        default_code = exp_out.get("code") or build_default_analysis_script(df_active, is_demo=is_demo)
        edited_code = st.text_area(
            "Review or Edit Python Analysis Script (`dataset.csv` is automatically mounted in the temp folder):",
            value=default_code,
            height=320,
        )
        exp_out["code"] = edited_code

        is_safe, safety_issues = validate_code_safety(edited_code)
        if is_safe:
            st.success("✅ Static AST Security Check Passed: All imports and calls match the scientific allow-list.")
        else:
            st.error("🚫 Static AST Security Check Blocked Execution:\n- " + "\n- ".join(safety_issues))

        if st.button("▶️ Run Python Analysis in Restricted Sandbox", type="primary", disabled=not is_safe, use_container_width=True):
            with st.spinner("Executing Python script in isolated sandbox subprocess..."):
                exec_res = execute_sandboxed_python(edited_code, df=df_active, timeout_sec=25)
                exp_out["execution"] = exec_res
                exp_out["is_demo"] = is_demo
                st.session_state["experiment_output"] = exp_out

                if exec_res["success"]:
                    try:
                        llm = get_active_llm()
                        tables_txt = "\n\n".join(
                            f"Table {t['name']}:\n{t['df'].to_string(index=False)}" for t in exec_res.get("tables", [])
                        )
                        methods_md = interpret_executed_results(
                            llm=llm,
                            research_question=rq,
                            executed_stdout=exec_res["stdout"],
                            table_summaries=tables_txt,
                            is_demo=is_demo,
                            provider_name=st.session_state.get("llm_provider", ""),
                            base_url=st.session_state.get("llm_base_url"),
                        )
                        exp_out["methods_results_md"] = methods_md
                        st.session_state["experiment_output"] = exp_out
                    except Exception as e:
                        st.warning(f"Code executed successfully; LLM narrative note: {e}")

    # Display Executed Outputs (stdout, tables, figures, methods draft)
    exec_data = exp_out.get("execution")
    if exec_data:
        if exp_out.get("is_demo"):
            st.warning("⚠️ **DEMO DATA – not real:** All numbers and figures below were computed on the synthetic demo dataset.")
        if exec_data.get("success"):
            st.success("✅ Subprocess Execution Succeeded (Exit Code 0)")
        else:
            st.error("❌ Subprocess Execution Error")

        if exec_data.get("stdout"):
            st.markdown("#### 🖥️ Executed `stdout` (Exact Statistical Output)")
            st.code(exec_data["stdout"], language="text")
        if exec_data.get("stderr"):
            with st.expander("⚠️ Subprocess `stderr` / Warnings"):
                st.code(exec_data["stderr"], language="text")

        if exec_data.get("tables"):
            st.markdown("#### 📊 Generated Statistical Tables")
            for tbl in exec_data["tables"]:
                st.markdown(f"**`{tbl['name']}`**")
                st.dataframe(tbl["df"], use_container_width=True)

        if exec_data.get("figures"):
            st.markdown("#### 📈 Generated Publication Figures")
            fig_cols = st.columns(min(2, len(exec_data["figures"])))
            for idx, fig_item in enumerate(exec_data["figures"]):
                with fig_cols[idx % len(fig_cols)]:
                    st.image(fig_item["bytes"], caption=fig_item["name"], use_container_width=True)

        if exp_out.get("methods_results_md"):
            st.markdown("#### 📝 Draft Methods & Empirical Results Section (Grounded in Executed Output)")
            st.markdown(exp_out["methods_results_md"])

    # Combine design + methods/results for download
    combined_exp_md = "\n\n---\n\n".join(
        part for part in [exp_out.get("design_md", ""), exp_out.get("methods_results_md", "")] if part
    )
    if combined_exp_md:
        render_download_toolbar(
            title=f"Experimental Design & Executed Analysis: {rq[:70]}",
            content_md=combined_exp_md,
            file_prefix="experiment_analysis_report",
            figures=(exec_data or {}).get("figures"),
            code_py=exp_out.get("code"),
        )

    # Biopython Bioinformatics Tool
    st.markdown("---")
    with st.expander("🧬 Bioinformatics Sequence Analyzer (`biopython` — DNA/RNA/Protein Statistics)"):
        fasta_input = st.text_area(
            "Paste FASTA or raw nucleotide sequence(s):",
            value=">TaDREB2_Wheat_Drought_Transcription_Factor\nATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGCGATCGATCGGTACGATCGATCG\n>TaWRKY19_Stress_Responsive_Gene\nATGTCGATCGATCGATCGATCGGCTAGCTAGCTAGCTAGCATCGATCGATCG",
            height=110,
        )
        if st.button("🔬 Compute Biopython Sequence Statistics"):
            bio_res = run_bioinformatics_sequence_stats(fasta_input)
            if not bio_res["summary_df"].empty:
                st.dataframe(bio_res["summary_df"], use_container_width=True)


# ===========================================================================
# PAGE 4: AI ACADEMIC WRITING & PUBLICATION AGENT
# ===========================================================================

def page_writing() -> None:
    st.markdown("## 🟣 Agent 4: AI Academic Writing & Publication Agent")
    st.caption(
        "“From data to publication.” — Generates a complete, canonically written academic manuscript "
        "with verified in-text citations, full bibliography, journal recommendations, and peer-review simulation."
    )

    refs = st.session_state.get("verified_references", [])
    exp_out = st.session_state.get("experiment_output") or {}
    lit_out = st.session_state.get("literature_output") or {}
    disc_out = st.session_state.get("discovery_output") or {}

    st.info(
        f"📚 **Verified Reference Library Ready:** `{len(refs)}` real API/PDF papers available for in-text citation. "
        f"| 🧪 **Executed Experiment Results Ready:** `{'Yes' if (exp_out.get('execution') or {}).get('stdout') else 'No (will use pre-registered placeholders or click Auto-Populate below)'}`"
    )

    if len(refs) == 0:
        if st.button("⚡ Quick-Fetch Verified References from OpenAlex & Crossref for Current Field"):
            from utils.search_apis import search_all_databases
            q = st.session_state.get("researcher_field", "AI drought prediction wheat")
            sr = search_all_databases(q, max_per_db=5)
            st.session_state["verified_references"] = merge_into_library([], sr["papers"])
            st.rerun()

    tab_ms, tab_jn, tab_pr = st.tabs(
        [
            "📝 1. Write Complete Canonical Manuscript & Cover Letter",
            "🏛️ 2. Journal Recommendations & Predatory Journal Checklist",
            "🧐 3. Simulated Peer Review & Response-to-Reviewers Letter",
        ]
    )

    with tab_ms:
        with st.form("manuscript_form"):
            default_title = (
                "Explainable Multi-Modal Machine Learning for Early Drought Stress and Grain Yield Prediction in Wheat (Triticum aestivum)"
            )
            paper_title = st.text_input("Canonical Manuscript Title", value=default_title)
            rq_val = st.text_input(
                "Core Research Question",
                value=lit_out.get("topic_query") or disc_out.get("query") or st.session_state.get("researcher_field", ""),
            )
            c1, c2 = st.columns(2)
            with c1:
                citation_style = st.selectbox(
                    "Citation Style",
                    options=["APA", "IEEE", "Harvard", "MLA", "Vancouver"],
                    index=["APA", "IEEE", "Harvard", "MLA", "Vancouver"].index(
                        st.session_state.get("citation_style", "APA")
                    ),
                )
            with c2:
                target_journal = st.text_input(
                    "Target Journal",
                    value=st.session_state.get("target_journal", "Computers and Electronics in Agriculture"),
                )
            author_guidelines = st.text_area(
                "Paste or Describe Target Journal Author Guidelines",
                value="Standard IMRaD structure (Introduction, Literature Review, Materials & Methods, Results, Discussion, Limitations, Conclusion). Include a 250-word structured abstract, 5 bullet highlights, keywords, CRediT authorship statement, AI-use disclosure, and full reference list with DOIs.",
                height=85,
            )
            gen_paper_btn = st.form_submit_button(
                "🖋️ Write Complete Canonical Paper (with Verified Citations & Bibliography)",
                use_container_width=True,
            )

        if gen_paper_btn:
            try:
                llm = get_active_llm()
                st.session_state["citation_style"] = citation_style
                st.session_state["target_journal"] = target_journal

                # Ensure we have verified references so the complete paper always includes real references
                current_refs = st.session_state.get("verified_references", [])
                if not current_refs:
                    from utils.search_apis import search_all_databases
                    sr = search_all_databases(f"{st.session_state.get('researcher_field', '')} {rq_val}", max_per_db=5)
                    current_refs = merge_into_library([], sr["papers"])
                    st.session_state["verified_references"] = current_refs

                exec_stdout = ((exp_out.get("execution") or {}).get("stdout")) or ""
                with st.spinner("Writing complete canonical manuscript (Abstract, Introduction, Literature, Methods, Results, Discussion, Declarations & Verified References)..."):
                    wrt_res = write_canonical_manuscript(
                        llm=llm,
                        paper_title=paper_title,
                        research_question=rq_val,
                        field=st.session_state.get("researcher_field", ""),
                        citation_style=citation_style,
                        target_journal=target_journal,
                        author_guidelines=author_guidelines,
                        verified_references=current_refs,
                        literature_context=lit_out.get("review_md"),
                        experiment_design_context=exp_out.get("design_md"),
                        executed_results_stdout=exec_stdout,
                        is_demo_data=bool(exp_out.get("is_demo", True)),
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    cur_w = st.session_state.get("writing_output") or {}
                    cur_w.update(wrt_res)
                    st.session_state["writing_output"] = cur_w
                    st.success("Canonical manuscript and submission cover letter generated!")
            except Exception as e:
                st.error(str(e))

        wrt_out = st.session_state.get("writing_output") or {}
        if wrt_out.get("manuscript_md"):
            st.markdown("### 📄 Complete Canonical Manuscript")
            st.markdown(wrt_out["manuscript_md"])

            render_download_toolbar(
                title=wrt_out.get("paper_title", "Academic Manuscript"),
                content_md=wrt_out["manuscript_md"],
                file_prefix="complete_canonical_paper",
                references=st.session_state.get("verified_references", []),
                figures=((exp_out.get("execution") or {}).get("figures")),
            )

            with st.expander("✉️ View & Download Journal Submission Cover Letter"):
                st.markdown(wrt_out.get("cover_letter_md", ""))
                render_download_toolbar(
                    title=f"Cover Letter — {wrt_out.get('paper_title', '')}",
                    content_md=wrt_out.get("cover_letter_md", ""),
                    file_prefix="submission_cover_letter",
                    references=[],
                )

    with tab_jn:
        st.markdown("### 🏛️ Journal Scope-Fit Finder (Live OpenAlex Sources & Crossref Journals)")
        st.caption("Never fabricates Impact Factors or acceptance rates — always verify current metrics on the official journal website.")
        jn_query = st.text_input(
            "Topic / Keywords for Journal Matching",
            value=st.session_state.get("researcher_field", "Agricultural AI crop modeling"),
        )
        if st.button("🔎 Recommend Suitable Peer-Reviewed Journals", use_container_width=True):
            try:
                llm = get_active_llm()
                with st.spinner("Querying OpenAlex Sources & Crossref Journals and analyzing scope fit..."):
                    jn_res = recommend_journals_with_fit(
                        llm=llm,
                        topic=jn_query,
                        field=st.session_state.get("researcher_field", ""),
                        target_journal_hint=st.session_state.get("target_journal", ""),
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    cur_w = st.session_state.get("writing_output") or {}
                    cur_w.update(jn_res)
                    st.session_state["writing_output"] = cur_w
            except Exception as e:
                st.error(str(e))

        wrt_out = st.session_state.get("writing_output") or {}
        if "journals_df" in wrt_out:
            st.dataframe(wrt_out["journals_df"], use_container_width=True)
            st.markdown(wrt_out.get("journal_analysis_md", ""))

        st.markdown("### 🛡️ Predatory-Journal Warning & Verification Checklist")
        st.dataframe(pd.DataFrame(PREDATORY_JOURNAL_CHECKLIST), use_container_width=True)

    with tab_pr:
        st.markdown("### 🧐 Simulated Peer Review (2 Reviewers + Associate Editor) & Rebuttal Generator")
        wrt_out = st.session_state.get("writing_output") or {}
        ms_text = wrt_out.get("manuscript_md", "")
        if not ms_text:
            st.info("Generate a manuscript in Tab 1 first (or paste your draft below) to run simulated peer review.")
        custom_draft = st.text_area(
            "Manuscript Text to Review",
            value=ms_text,
            height=220,
        )
        rev_notes = st.text_input(
            "Optional Author Notes for the Point-by-Point Rebuttal Letter",
            value="We will add an external validation cohort, clarify hyperparameter tuning, and expand the limitations section.",
        )
        if st.button("🧪 Run Double-Anonymous Peer Review Simulation & Draft Rebuttal Letter", use_container_width=True):
            if not custom_draft.strip():
                st.warning("Please generate or paste a manuscript draft first.")
            else:
                try:
                    llm = get_active_llm()
                    with st.spinner("Simulating Reviewer 1, Reviewer 2, Associate Editor Decision & Point-by-Point Response Letter..."):
                        pr_res = simulate_peer_review_and_rebuttal(
                            llm=llm,
                            manuscript_md=custom_draft,
                            target_journal=st.session_state.get("target_journal", "Peer-Reviewed Journal"),
                            author_revision_notes=rev_notes,
                            provider_name=st.session_state.get("llm_provider", ""),
                            base_url=st.session_state.get("llm_base_url"),
                        )
                        cur_w = st.session_state.get("writing_output") or {}
                        cur_w.update(pr_res)
                        st.session_state["writing_output"] = cur_w
                except Exception as e:
                    st.error(str(e))

        wrt_out = st.session_state.get("writing_output") or {}
        if wrt_out.get("peer_review_md"):
            st.markdown("#### 📋 Simulated Peer Review Report")
            st.markdown(wrt_out["peer_review_md"])
            st.markdown("#### ✉️ Point-by-Point Response to Reviewers Letter")
            st.markdown(wrt_out.get("rebuttal_md", ""))
            render_download_toolbar(
                title="Simulated Peer Review & Point-by-Point Response Letter",
                content_md=wrt_out["peer_review_md"] + "\n\n---\n\n" + wrt_out.get("rebuttal_md", ""),
                file_prefix="peer_review_and_rebuttal",
                references=[],
            )


# ===========================================================================
# PAGE 5: AI PHD SUPERVISOR AGENT
# ===========================================================================

def page_supervisor() -> None:
    st.markdown("## 🩵 Agent 5: AI PhD Supervisor Agent")
    st.caption("“Your always-available research mentor.” — Plan, Guide, Feedback, Improve, Succeed.")

    # Interactive Milestone Checklist
    st.markdown("### ✅ PhD & Research Milestone Progress Tracker")
    checks = st.session_state.get("supervisor_milestone_checks", {})
    chk_cols = st.columns(len(checks))
    done_count = 0
    for idx, (m_label, m_val) in enumerate(list(checks.items())):
        with chk_cols[idx]:
            new_val = st.checkbox(m_label, value=m_val, key=f"sup_chk_{idx}")
            checks[m_label] = new_val
            if new_val:
                done_count += 1
    st.session_state["supervisor_milestone_checks"] = checks
    st.progress(done_count / max(1, len(checks)), text=f"Milestone Progress: {done_count} / {len(checks)} Pillars Completed")

    tab_road, tab_def, tab_eth, tab_chat = st.tabs(
        [
            "🗺️ 1. Personalized Roadmap & Critical Review Mode",
            "🎤 2. Slides, Poster & Mock Viva Defense Prep",
            "⚖️ 3. Research Ethics & Responsible AI Academy",
            "💬 4. 24/7 AI Supervisor Chat (with Memory)",
        ]
    )

    with tab_road:
        c1, c2 = st.columns(2)
        with c1:
            sup_topic = st.text_input(
                "Thesis / Project Topic",
                value=(st.session_state.get("writing_output") or {}).get("paper_title")
                or st.session_state.get("workflow_topic", "AI-based drought prediction in wheat"),
            )
            duration = st.selectbox(
                "Program Horizon",
                options=[
                    "1-Semester BS Capstone / Honors Thesis (4–6 Months)",
                    "2-Year Master's Thesis (24 Months)",
                    "3-Year PhD Journey (36 Months)",
                    "4-Year PhD Journey (48 Months)",
                    "12-Month Industry R&D / Postdoc Fellowship",
                ],
                index=2,
            )
        with c2:
            assumptions = st.text_area(
                "Your Current Assumptions / Methodology Plan (for Friendly Critical Review Mode)",
                value="We assume multi-spectral satellite vegetation indices combined with soil moisture sensors can generalize across climatic zones without site-specific recalibration.",
                height=105,
            )

        if st.button("🧭 Generate 5-Pillar Roadmap & Friendly Critical Review", use_container_width=True):
            try:
                llm = get_active_llm()
                with st.spinner("Crafting your personalized roadmap and Socratic critical review..."):
                    sup_res = generate_phd_roadmap_and_critique(
                        llm=llm,
                        research_topic=sup_topic,
                        field=st.session_state.get("researcher_field", ""),
                        career_stage=st.session_state.get("researcher_stage", "PhD Student"),
                        program_duration=duration,
                        current_draft_or_assumptions=assumptions,
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    cur_s = st.session_state.get("supervisor_output") or {}
                    cur_s.update(sup_res)
                    st.session_state["supervisor_output"] = cur_s
            except Exception as e:
                st.error(str(e))

        sup_out = st.session_state.get("supervisor_output") or {}
        if sup_out.get("roadmap_md"):
            st.markdown("#### 🗺️ Personalized Research Roadmap (Plan → Guide → Feedback → Improve → Succeed)")
            st.markdown(sup_out["roadmap_md"])
            st.markdown("#### 🤔 Friendly Critical Review Mode (Challenging Assumptions)")
            st.markdown(sup_out.get("critique_md", ""))
            render_download_toolbar(
                title=f"PhD Roadmap & Critical Supervisor Review: {sup_topic}",
                content_md=sup_out["roadmap_md"] + "\n\n---\n\n" + sup_out.get("critique_md", ""),
                file_prefix="phd_supervisor_roadmap",
                references=[],
            )

    with tab_def:
        st.markdown("### 🎤 Presentation Slides, Conference Poster & Mock Viva Defense Generator")
        if st.button("🎓 Generate 12-Slide Outline, 4-Quadrant Poster Layout & 10 Mock Viva Q&As", use_container_width=True):
            try:
                llm = get_active_llm()
                ms_ctx = (st.session_state.get("writing_output") or {}).get("manuscript_md", "")
                with st.spinner("Preparing presentation outline, poster layout, and mock viva questions with model answers..."):
                    def_md = generate_defense_and_presentation_kit(
                        llm=llm,
                        research_topic=st.session_state.get("workflow_topic", "AI in scientific research"),
                        field=st.session_state.get("researcher_field", ""),
                        manuscript_or_summary=ms_ctx,
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    cur_s = st.session_state.get("supervisor_output") or {}
                    cur_s["defense_kit_md"] = def_md
                    st.session_state["supervisor_output"] = cur_s
            except Exception as e:
                st.error(str(e))

        sup_out = st.session_state.get("supervisor_output") or {}
        if sup_out.get("defense_kit_md"):
            st.markdown(sup_out["defense_kit_md"])
            render_download_toolbar(
                title="Presentation, Poster & Viva Defense Kit",
                content_md=sup_out["defense_kit_md"],
                file_prefix="viva_defense_and_poster_kit",
                references=[],
            )

    with tab_eth:
        st.markdown("### ⚖️ Research Ethics & Responsible AI Curriculum")
        st.dataframe(pd.DataFrame(RESEARCH_ETHICS_MODULES), use_container_width=True)
        st.info(
            "💡 **Ready-to-Use AI Disclosure Template for Journals:**\n\n"
            "> *“During the preparation of this work, the author(s) used ResearchAI to assist with literature organization, "
            "analysis code drafting, and language editing. After using this tool, the author(s) reviewed, verified, and "
            "edited all content, citations, and statistical outputs and take full responsibility for the content of the publication.”*"
        )

    with tab_chat:
        st.markdown("### 💬 24/7 Conversational Academic Mentor (Streaming with Memory)")
        history = st.session_state.get("supervisor_chat_history", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask your AI PhD Supervisor about methodology, thesis writing, reviewer comments, or defense prep...")
        if user_q:
            history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                try:
                    llm = get_active_llm()
                    profile = {
                        "field": st.session_state.get("researcher_field", ""),
                        "stage": st.session_state.get("researcher_stage", ""),
                        "target_journal": st.session_state.get("target_journal", ""),
                    }
                    proj_ctx = ((st.session_state.get("writing_output") or {}).get("manuscript_md") or "")[:1500]
                    stream_gen = stream_supervisor_chat(
                        llm=llm,
                        user_message=user_q,
                        chat_history=history[:-1],
                        researcher_profile=profile,
                        project_context=proj_ctx,
                        provider_name=st.session_state.get("llm_provider", ""),
                        base_url=st.session_state.get("llm_base_url"),
                    )
                    full_reply = st.write_stream(stream_gen)
                    history.append({"role": "assistant", "content": str(full_reply)})
                    st.session_state["supervisor_chat_history"] = history
                except Exception as e:
                    st.error(str(e))


# ===========================================================================
# PAGE 6: FULL WORKFLOW (LANGGRAPH) & STEP 6 "CREATE IMPACT"
# ===========================================================================

def page_full_workflow() -> None:
    st.markdown("## 🚀 Complete LangGraph Research Pipeline & Step 6: Create Impact")
    st.caption(
        "Pipeline: **1. Discover Ideas → 2. Review Literature → 3. Design & Analyze → "
        "4. Write & Publish → 5. Get Mentored → 6. Create Impact** (with Human-Approval Checkpoints)"
    )

    wf_topic = st.text_input(
        "Master Research Pipeline Topic",
        value=st.session_state.get("workflow_topic", "AI-based drought stress prediction in wheat using multi-spectral UAV & soil sensors"),
    )
    st.session_state["workflow_topic"] = wf_topic

    step_labels = {
        1: "Step 1: Discover Ideas (Discovery Agent)",
        2: "Step 2: Review Literature (Literature Agent + Verified References)",
        3: "Step 3: Design & Analyze (Experiment Agent + Safe Subprocess Code Execution)",
        4: "Step 4: Write & Publish (Canonical Manuscript + Verified Citations + Journal Fit)",
        5: "Step 5: Get Mentored (PhD Supervisor Roadmap + Critical Review)",
        6: "Step 6: Create Impact (Lay Summary, LinkedIn Post, Policy Brief, Press Release & Visibility Plan)",
    }

    current_step = int(st.session_state.get("workflow_current_step", 1))
    st.markdown(f"### 🚦 Human-in-the-Loop Checkpoint Controller — Current Stage: **{step_labels.get(current_step, 'Completed')}**")

    c_step, c_approve, c_all, c_reset = st.columns(4)

    def _build_wf_state(stop_step: int) -> ResearchWorkflowState:
        return {
            "provider": st.session_state.get("llm_provider", "OpenAI"),
            "model": st.session_state.get("llm_model", "gpt-4o-mini"),
            "api_key": st.session_state.get("llm_api_key") or None,
            "base_url": st.session_state.get("llm_base_url") or None,
            "temperature": float(st.session_state.get("llm_temperature", 0.3)),
            "max_tokens": int(st.session_state.get("llm_max_tokens", 2800)),
            "field": st.session_state.get("researcher_field", "Agricultural & Environmental AI"),
            "topic": wf_topic,
            "career_stage": st.session_state.get("researcher_stage", "PhD Student"),
            "available_resources": st.session_state.get("available_resources", "University compute & open datasets"),
            "citation_style": st.session_state.get("citation_style", "APA"),
            "target_journal": st.session_state.get("target_journal", "Computers and Electronics in Agriculture"),
            "stop_after_step": stop_step,
            "verified_references": st.session_state.get("verified_references", []),
            "discovery_output": st.session_state.get("discovery_output"),
            "literature_output": st.session_state.get("literature_output"),
            "experiment_output": st.session_state.get("experiment_output"),
            "writing_output": st.session_state.get("writing_output"),
            "supervisor_output": st.session_state.get("supervisor_output"),
            "impact_output": st.session_state.get("impact_output"),
            "completed_steps": st.session_state.get("workflow_approved_steps", []),
        }

    def _sync_state_back(updated: Dict[str, Any]) -> None:
        for key in (
            "verified_references",
            "discovery_output",
            "literature_output",
            "experiment_output",
            "writing_output",
            "supervisor_output",
            "impact_output",
        ):
            if updated.get(key) is not None:
                st.session_state[key] = updated[key]
        if updated.get("completed_steps") is not None:
            st.session_state["workflow_approved_steps"] = updated["completed_steps"]

    with c_step:
        if st.button(f"▶️ Run {step_labels.get(current_step, 'Step')[:18]}...", use_container_width=True):
            try:
                with st.spinner(f"Executing {step_labels.get(current_step)} via LangGraph node..."):
                    state_in = _build_wf_state(current_step)
                    delta = run_single_workflow_stage(current_step, state_in)
                    _sync_state_back(delta)
                    st.success(f"Completed {step_labels.get(current_step)}! Review output below and click 'Approve & Advance'.")
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    with c_approve:
        if st.button("✅ Approve Checkpoint & Advance to Next Step", use_container_width=True):
            if current_step < 6:
                st.session_state["workflow_current_step"] = current_step + 1
                st.rerun()
            else:
                st.success("All 6 steps approved and complete!")

    with c_all:
        if st.button("⚡ Run Full 6-Step LangGraph Pipeline", use_container_width=True):
            try:
                with st.spinner("Running complete 6-node LangGraph `StateGraph` (Discover -> Literature -> Experiment -> Writing -> Supervisor -> Impact)..."):
                    graph = build_research_workflow_graph()
                    final_state = graph.invoke(_build_wf_state(stop_step=6))
                    _sync_state_back(final_state)
                    st.session_state["workflow_current_step"] = 6
                    st.success("Full 6-Step LangGraph Pipeline executed successfully!")
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    with c_reset:
        if st.button("🔄 Reset Step Counter to 1", use_container_width=True):
            st.session_state["workflow_current_step"] = 1
            st.rerun()

    st.markdown("---")
    st.markdown("### 🌍 Step 6: Create Impact (Standalone or Pipeline Generator)")
    st.caption(
        "Produces: Plain-Language Summary, Social/LinkedIn Post, Policy/Industry Brief, "
        "Press-Release Summary, Visibility Plan (ORCID, Google Scholar, Preprints, Conferences) & Citation Tracking Checklist."
    )
    if st.button("📣 Generate / Refresh Step 6 Impact & Outreach Package", use_container_width=True):
        try:
            llm = get_active_llm()
            ms_md = (st.session_state.get("writing_output") or {}).get("manuscript_md", "")
            with st.spinner("Generating plain-language summary, LinkedIn post, policy brief, press release & visibility plan..."):
                imp = generate_impact_package(
                    llm=llm,
                    topic=wf_topic,
                    field=st.session_state.get("researcher_field", ""),
                    manuscript_md=ms_md,
                    provider_name=st.session_state.get("llm_provider", ""),
                    base_url=st.session_state.get("llm_base_url"),
                )
                st.session_state["impact_output"] = imp
                st.success("Step 6 Impact Package generated!")
        except Exception as e:
            st.error(str(e))

    imp_out = st.session_state.get("impact_output")
    if imp_out and imp_out.get("impact_md"):
        st.markdown(imp_out["impact_md"])
        render_download_toolbar(
            title=f"Step 6 Research Impact & Outreach Package: {wf_topic}",
            content_md=imp_out["impact_md"],
            file_prefix="research_impact_package",
            references=st.session_state.get("verified_references", []),
        )


# ===========================================================================
# PAGE 7: PROJECT TRACKER
# ===========================================================================

def page_project_tracker() -> None:
    st.markdown("## 📋 Research Project Tracker Board")
    st.caption("Track all your projects across stages: **Idea → Literature → Experiment → Writing → Submitted → Revision → Published**.")

    stages = ["Idea", "Literature", "Experiment", "Writing", "Submitted", "Revision", "Published"]
    board: List[Dict[str, str]] = st.session_state.get("projects_board", [])

    with st.expander("➕ Add New Research Project to Tracker", expanded=False):
        with st.form("add_project_form"):
            p_title = st.text_input("Project Title", placeholder="e.g., Physics-Informed Neural Networks for Soil Moisture Forecasting")
            c1, c2, c3 = st.columns(3)
            with c1:
                p_field = st.text_input("Field", value=st.session_state.get("researcher_field", ""))
            with c2:
                p_stage = st.selectbox("Current Status", options=stages, index=0)
            with c3:
                p_journal = st.text_input("Target Journal", value=st.session_state.get("target_journal", ""))
            p_notes = st.text_area("Milestones / Next Action Notes", height=70)
            if st.form_submit_button("Add Project"):
                if p_title.strip():
                    board.append(
                        {
                            "id": f"PROJ-{len(board) + 1:02d}",
                            "title": p_title.strip(),
                            "field": p_field.strip(),
                            "stage": p_stage,
                            "target_journal": p_journal.strip(),
                            "notes": p_notes.strip(),
                        }
                    )
                    st.session_state["projects_board"] = board
                    st.success(f"Added `{p_title.strip()}` to Project Tracker.")
                    st.rerun()

    # Status metrics across the 7 stages
    cols = st.columns(len(stages))
    for idx, stg in enumerate(stages):
        cnt = sum(1 for p in board if p.get("stage") == stg)
        cols[idx].metric(stg, cnt)

    # Interactive status editor per project
    for idx, proj in enumerate(board):
        with st.container():
            c_id, c_t, c_s, c_n = st.columns([1, 3, 2, 3])
            c_id.markdown(f"**{proj['id']}**")
            c_t.markdown(f"**{proj['title']}**  \n*{proj['field']}* &bull; Target: `{proj['target_journal']}`")
            new_stg = c_s.selectbox(
                "Stage",
                options=stages,
                index=stages.index(proj["stage"]) if proj["stage"] in stages else 0,
                key=f"proj_stg_{idx}",
                label_visibility="collapsed",
            )
            board[idx]["stage"] = new_stg
            new_note = c_n.text_input("Notes", value=proj.get("notes", ""), key=f"proj_note_{idx}", label_visibility="collapsed")
            board[idx]["notes"] = new_note

    st.session_state["projects_board"] = board

    df_board = pd.DataFrame(board)
    st.markdown("### 📊 Master Portfolio Table")
    st.dataframe(df_board, use_container_width=True)

    summary_md = "# ResearchAI Project Tracker Summary\n\n"
    for p in board:
        summary_md += (
            f"## {p['id']}: {p['title']}\n"
            f"- **Stage**: {p['stage']}\n"
            f"- **Field**: {p['field']}\n"
            f"- **Target Journal**: {p['target_journal']}\n"
            f"- **Notes**: {p['notes']}\n\n"
        )
    render_download_toolbar(
        title="ResearchAI Portfolio & Project Tracker Summary",
        content_md=summary_md,
        file_prefix="project_tracker_summary",
        references=[],
    )


# ===========================================================================
# PAGE 8: EXPORT CENTER
# ===========================================================================

def page_export_center() -> None:
    st.markdown("## 📦 Unified Manuscript & Research Dossier Export Center")
    st.caption(
        "Download your **Complete Canonical Paper** (with Verified References) or your **Full 6-Step Research Dossier** "
        "in Microsoft Word (`.docx`), PDF (`.pdf`), Markdown (`.md`), and reproducible Python (`.py`)."
    )

    wrt_out = st.session_state.get("writing_output") or {}
    disc_out = st.session_state.get("discovery_output") or {}
    lit_out = st.session_state.get("literature_output") or {}
    exp_out = st.session_state.get("experiment_output") or {}
    sup_out = st.session_state.get("supervisor_output") or {}
    imp_out = st.session_state.get("impact_output") or {}
    refs = st.session_state.get("verified_references", [])

    st.markdown("### 1. Complete Canonical Research Paper (Manuscript + Verified References)")
    if wrt_out.get("manuscript_md"):
        render_download_toolbar(
            title=wrt_out.get("paper_title", "Canonical Academic Manuscript"),
            content_md=wrt_out["manuscript_md"],
            file_prefix="ResearchAI_Canonical_Paper",
            references=refs,
            figures=((exp_out.get("execution") or {}).get("figures")),
            code_py=exp_out.get("code"),
        )
    else:
        st.info("No manuscript drafted yet in this session. Go to **4. Writing & Publication** or **Full Workflow** to generate your complete canonical paper.")

    st.markdown("---")
    st.markdown("### 2. Complete End-to-End Research Dossier (All Completed Agents Combined)")
    sections = []
    if disc_out.get("report_md"):
        sections.append("# PART I: RESEARCH DISCOVERY & 5 NOVEL IDEAS\n\n" + disc_out["report_md"])
    if lit_out.get("review_md"):
        sections.append("# PART II: PRISMA SYSTEMATIC LITERATURE REVIEW\n\n" + lit_out["review_md"])
    if exp_out.get("design_md") or exp_out.get("methods_results_md"):
        sections.append(
            "# PART III: EXPERIMENTAL DESIGN & EXECUTED STATISTICAL ANALYSIS\n\n"
            + (exp_out.get("design_md") or "")
            + "\n\n"
            + (exp_out.get("methods_results_md") or "")
        )
    if wrt_out.get("manuscript_md"):
        sections.append("# PART IV: CANONICAL ACADEMIC MANUSCRIPT\n\n" + wrt_out["manuscript_md"])
    if sup_out.get("roadmap_md"):
        sections.append("# PART V: PHD SUPERVISOR ROADMAP & CRITICAL REVIEW\n\n" + sup_out["roadmap_md"])
    if imp_out.get("impact_md"):
        sections.append("# PART VI: RESEARCH IMPACT & OUTREACH PACKAGE\n\n" + imp_out["impact_md"])

    if sections:
        full_dossier_md = "\n\n---\n\n".join(sections)
        render_download_toolbar(
            title="ResearchAI Complete Multi-Agent Research Dossier",
            content_md=full_dossier_md,
            file_prefix="ResearchAI_Complete_Dossier",
            references=refs,
            figures=((exp_out.get("execution") or {}).get("figures")),
            code_py=exp_out.get("code"),
        )
    else:
        st.info("Run any agent to populate your downloadable multi-agent research dossier.")


# ===========================================================================
# PAGE 9: USER GUIDE & ETHICS
# ===========================================================================

def page_guide() -> None:
    st.markdown("## 📘 ResearchAI User Guide & Responsible AI in Research")
    guide_path = Path(__file__).resolve().parent / "GUIDE.md"
    if guide_path.exists():
        st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.info("GUIDE.md documentation file will appear here.")


# ---------------------------------------------------------------------------
# Router Dispatch
# ---------------------------------------------------------------------------

page_dispatch = {
    "Home": page_home,
    "1. Discovery Agent": page_discovery,
    "2. Literature Agent": page_literature,
    "3. Experiment Agent": page_experiment,
    "4. Writing & Publication": page_writing,
    "5. PhD Supervisor": page_supervisor,
    "Full Workflow & Impact": page_full_workflow,
    "Project Tracker": page_project_tracker,
    "Export Center": page_export_center,
    "User Guide & Ethics": page_guide,
}

page_fn = page_dispatch.get(active_page, page_home)
page_fn()
render_footer()
