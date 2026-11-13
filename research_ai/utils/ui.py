"""
utils/ui.py
===========
Shared Streamlit UI components, session-state initialization, sidebar LLM + Researcher
Profile configuration, Hero Header with Active Provider Badge, 6-Step Progress Stepper,
Academic Disclaimer banner, and Multi-Format Download Toolbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.exporters import export_to_docx, export_to_markdown, export_to_pdf
from utils.llm_factory import (
    CUSTOM_API_PRESETS,
    PROVIDER_MODELS,
    get_llm,
    resolve_credentials,
    test_llm_connection,
    validate_base_url,
)


def init_session_state() -> None:
    """Initialize all required session state keys for standalone & pipeline operation."""
    defaults: Dict[str, Any] = {
        # Navigation
        "active_page": "Home",
        # LLM Settings
        "llm_provider": "OpenAI",
        "llm_model": "gpt-4o-mini",
        "llm_custom_model_toggle": False,
        "llm_api_key": "",
        "llm_base_url": "",
        "llm_temperature": 0.3,
        "llm_max_tokens": 3000,
        "llm_extra_headers_json": "",
        "custom_preset_name": "OpenRouter (Cloud — Free & Paid Models)",
        # Saved custom API profiles (up to 3, stored in st.session_state only — never on disk)
        "saved_custom_profiles": {},
        # Researcher Profile
        "researcher_field": "Agricultural & Environmental AI (Crop Drought Prediction)",
        "researcher_stage": "PhD Student",
        "citation_style": "APA",
        "target_journal": "Computers and Electronics in Agriculture / Nature Food",
        "available_resources": "Standard university lab workstation, public satellite/climate datasets, field trial plots",
        # Verified Reference Library (ONLY real API/PDF papers with DOIs/links)
        "verified_references": [],
        # Agent Outputs (shared context across pipeline)
        "discovery_output": None,
        "literature_output": None,
        "experiment_output": None,
        "writing_output": None,
        "supervisor_output": None,
        "impact_output": None,
        # Supervisor 24/7 Mentor Chat Memory
        "supervisor_chat_history": [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I am your **AI PhD Supervisor & Research Mentor**. "
                    "Ask me anything about your research question, methodology, thesis milestones, "
                    "viva/defense preparation, or academic ethics (CRediT authorship, AI disclosure, data integrity)."
                ),
            }
        ],
        "supervisor_milestone_checks": {
            "Plan: Define RQ, gap & supervisor charter": True,
            "Guide: Complete PRISMA literature review & verified bibliography": False,
            "Feedback: Pre-register experimental protocol & power analysis": False,
            "Improve: Execute reproducible analysis & validate assumptions": False,
            "Succeed: Draft canonical manuscript, pass mock viva & submit": False,
        },
        # Project Tracker Board
        "projects_board": [
            {
                "id": "PROJ-01",
                "title": "AI-Based Multi-Modal Drought Prediction in Wheat (Triticum aestivum)",
                "field": "Agricultural AI & Crop Science",
                "stage": "Idea",
                "target_journal": "Computers and Electronics in Agriculture",
                "notes": "Initial flagship research project in ResearchAI.",
            }
        ],
        # Full LangGraph Workflow Checkpoint State
        "workflow_current_step": 1,
        "workflow_topic": "AI-based drought stress prediction in wheat using multi-spectral UAV & soil sensors",
        "workflow_approved_steps": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_custom_css() -> None:
    """Inject custom dark-navy + blue/teal CSS from assets/style.css."""
    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def get_active_llm():
    """Instantiate the active LangChain chat model using current sidebar settings."""
    extra_headers = None
    raw_headers = (st.session_state.get("llm_extra_headers_json") or "").strip()
    if raw_headers:
        try:
            import json
            extra_headers = json.loads(raw_headers)
        except Exception:
            extra_headers = None

    return get_llm(
        provider=st.session_state["llm_provider"],
        model=st.session_state["llm_model"],
        api_key=st.session_state.get("llm_api_key") or None,
        base_url=st.session_state.get("llm_base_url") or None,
        temperature=float(st.session_state.get("llm_temperature", 0.3)),
        max_tokens=int(st.session_state.get("llm_max_tokens", 3000)),
        extra_headers=extra_headers,
    )


def render_hero_header() -> None:
    """Render top Hero Banner with title, tagline, sub-tagline, and active provider badge."""
    provider = st.session_state.get("llm_provider", "OpenAI")
    model = st.session_state.get("llm_model", "gpt-4o-mini")
    creds = resolve_credentials(
        provider,
        st.session_state.get("llm_api_key"),
        st.session_state.get("llm_base_url"),
        model,
    )
    effective_model = creds.get("model") or model
    has_key = bool(creds.get("api_key"))
    status_label = "Key Ready" if has_key else "Key Needed in Sidebar"

    st.markdown(
        f"""
        <div class="rai-hero">
          <div>
            <h1 class="rai-hero-title">🔬 Research<span>AI</span></h1>
            <p class="rai-hero-tagline">Five AI Agents. One Smarter Research Journey.</p>
            <p class="rai-hero-subtagline">From idea to impact — your AI research team.</p>
          </div>
          <div>
            <div class="rai-provider-badge">
              <span class="rai-dot-live"></span>
              <span><strong>{provider}</strong> &bull; <code>{effective_model}</code> ({status_label})</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_stepper() -> None:
    """Render the 6-step research workflow progress stepper at the top of the app."""
    steps = [
        ("1. Discover Ideas", st.session_state.get("discovery_output") is not None, "1. Discovery Agent"),
        ("2. Review Literature", st.session_state.get("literature_output") is not None, "2. Literature Agent"),
        ("3. Design & Analyze", st.session_state.get("experiment_output") is not None, "3. Experiment Agent"),
        ("4. Write & Publish", st.session_state.get("writing_output") is not None, "4. Writing & Publication"),
        ("5. Get Mentored", st.session_state.get("supervisor_output") is not None, "5. PhD Supervisor"),
        ("6. Create Impact", st.session_state.get("impact_output") is not None, "Full Workflow & Impact"),
    ]
    active_page = st.session_state.get("active_page", "Home")
    html_parts = ['<div class="rai-stepper">']
    for label, is_done, page_match in steps:
        cls = "rai-step"
        if active_page == page_match:
            cls += " active"
        elif is_done:
            cls += " done"
        icon = "✅ " if is_done else ("🔹 " if active_page == page_match else "⚪ ")
        html_parts.append(f'<div class="{cls}">{icon}{label}</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_academic_disclaimer() -> None:
    """Render the mandatory visible academic verification & ethics disclaimer."""
    st.markdown(
        """
        <div class="rai-disclaimer">
          <strong>⚠️ Academic Integrity & Verification Notice:</strong>
          All research ideas, literature syntheses, statistical outputs, and manuscript drafts generated by
          <strong>ResearchAI</strong> must be independently verified by the researcher and approved by their academic supervisor.
          Citations are restricted to real records retrieved from scholarly APIs (arXiv, OpenAlex, Semantic Scholar, Crossref, PubMed)
          or uploaded PDFs. The human researcher remains solely accountable for originality, ethics, AI-use disclosure, and journal policy compliance.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the canonical footer motto."""
    st.markdown(
        """
        <div class="rai-footer">
          <div class="rai-footer-motto">“Higher Research, Brighter Tomorrow.”</div>
          <div>ResearchAI &mdash; Five AI Agents. One Smarter Research Journey. &bull; Built with LangGraph, LangChain &amp; Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_toolbar(
    title: str,
    content_md: str,
    file_prefix: str,
    references: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    code_py: Optional[str] = None,
) -> None:
    """Render one-click download buttons for Markdown (.md), Word (.docx), PDF (.pdf), and optional Python (.py)."""
    if not content_md:
        return
    refs = references if references is not None else st.session_state.get("verified_references", [])
    citation_style = st.session_state.get("citation_style", "APA")
    metadata = {
        "field": st.session_state.get("researcher_field", ""),
        "career_stage": st.session_state.get("researcher_stage", ""),
        "target_journal": st.session_state.get("target_journal", ""),
    }

    cols = st.columns(4 if code_py else 3)
    md_bytes = export_to_markdown(title, content_md, references=refs, citation_style=citation_style, metadata=metadata)
    docx_bytes = export_to_docx(title, content_md, references=refs, citation_style=citation_style, metadata=metadata, figures=figures)
    pdf_bytes = export_to_pdf(title, content_md, references=refs, citation_style=citation_style, metadata=metadata)

    with cols[0]:
        st.download_button(
            "📥 Download Markdown (.md)",
            data=md_bytes,
            file_name=f"{file_prefix}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{file_prefix}_{ abs(hash(title)) % 10000 }",
        )
    with cols[1]:
        st.download_button(
            "📄 Download Word (.docx)",
            data=docx_bytes,
            file_name=f"{file_prefix}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key=f"dl_docx_{file_prefix}_{ abs(hash(title)) % 10000 }",
        )
    with cols[2]:
        st.download_button(
            "📕 Download PDF (.pdf)",
            data=pdf_bytes,
            file_name=f"{file_prefix}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_pdf_{file_prefix}_{ abs(hash(title)) % 10000 }",
        )
    if code_py and len(cols) == 4:
        with cols[3]:
            st.download_button(
                "🐍 Download Script (.py)",
                data=code_py.encode("utf-8"),
                file_name=f"{file_prefix}_analysis.py",
                mime="text/x-python",
                use_container_width=True,
                key=f"dl_py_{file_prefix}_{ abs(hash(title)) % 10000 }",
            )


def render_sidebar() -> str:
    """
    Render the complete Sidebar:
    1. Navigation Menu
    2. Multi-Provider LLM Settings + Custom API Presets + 3 Session Profiles + Test Connection
    3. Researcher Profile (Field, Career Stage, Citation Style, Target Journal)
    4. Verified Reference Library Counter
    Returns the selected navigation page string.
    """
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        pages = [
            "Home",
            "1. Discovery Agent",
            "2. Literature Agent",
            "3. Experiment Agent",
            "4. Writing & Publication",
            "5. PhD Supervisor",
            "Full Workflow & Impact",
            "Project Tracker",
            "Export Center",
            "User Guide & Ethics",
        ]
        current_idx = pages.index(st.session_state["active_page"]) if st.session_state["active_page"] in pages else 0
        selected_page = st.radio(
            "Go to Section:",
            options=pages,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state["active_page"] = selected_page

        st.markdown("---")
        st.markdown("## 🤖 AI Provider & Model")
        st.caption(
            "💡 **Free options available:** Groq (free tier), Google Gemini (free tier), "
            "OpenRouter (`:free` models), or Ollama / LM Studio running locally!"
        )

        provider_options = list(PROVIDER_MODELS.keys())
        prov_idx = (
            provider_options.index(st.session_state["llm_provider"])
            if st.session_state["llm_provider"] in provider_options
            else 0
        )
        provider = st.selectbox("LLM Provider", options=provider_options, index=prov_idx)
        st.session_state["llm_provider"] = provider

        # Custom API Presets & Saved Profiles
        if provider == "Custom API (OpenAI-compatible)":
            preset_names = list(CUSTOM_API_PRESETS.keys())
            preset_choice = st.selectbox(
                "Quick Presets (Auto-Fill)",
                options=preset_names,
                index=preset_names.index(st.session_state.get("custom_preset_name", preset_names[0]))
                if st.session_state.get("custom_preset_name") in preset_names
                else 0,
            )
            if preset_choice != st.session_state.get("custom_preset_name"):
                st.session_state["custom_preset_name"] = preset_choice
                preset_cfg = CUSTOM_API_PRESETS[preset_choice]
                st.session_state["llm_base_url"] = preset_cfg["base_url"]
                st.session_state["llm_model"] = preset_cfg["model"]
                if preset_cfg.get("extra_headers"):
                    import json
                    st.session_state["llm_extra_headers_json"] = json.dumps(preset_cfg["extra_headers"])
                else:
                    st.session_state["llm_extra_headers_json"] = ""

            if not st.session_state.get("llm_base_url"):
                st.session_state["llm_base_url"] = CUSTOM_API_PRESETS[preset_choice]["base_url"]

            st.caption(CUSTOM_API_PRESETS[preset_choice]["hint"])

            base_url_val = st.text_input(
                "Base URL",
                value=st.session_state.get("llm_base_url", "https://openrouter.ai/api/v1"),
                placeholder="https://openrouter.ai/api/v1 or http://localhost:11434/v1",
            )
            st.session_state["llm_base_url"] = base_url_val.strip()
            valid_url, url_hint = validate_base_url(st.session_state["llm_base_url"])
            if not valid_url:
                st.warning(url_hint)
            elif "Hint:" in url_hint or "Note:" in url_hint:
                st.info(url_hint)

            custom_model_val = st.text_input(
                "Model Name (Free Text)",
                value=st.session_state.get("llm_model", "meta-llama/llama-3.3-70b-instruct:free"),
                help="Enter any model ID supported by your endpoint (e.g. `deepseek-chat`, `llama3.2`, `meta-llama/llama-3.3-70b-instruct:free`).",
            )
            st.session_state["llm_model"] = custom_model_val.strip()

            with st.expander("Optional Extra Headers & Saved Profiles (up to 3)"):
                hdr_val = st.text_area(
                    "Extra Headers (JSON)",
                    value=st.session_state.get("llm_extra_headers_json", ""),
                    placeholder='{"HTTP-Referer": "https://my-lab.edu", "X-Title": "ResearchAI"}',
                    height=68,
                )
                st.session_state["llm_extra_headers_json"] = hdr_val.strip()

                # Save up to 3 named custom profiles in st.session_state only
                profiles: Dict[str, Dict[str, str]] = st.session_state.get("saved_custom_profiles", {})
                prof_name = st.text_input("Profile Name (Session Only)", placeholder="e.g., My OpenRouter Free")
                c_save, c_load = st.columns(2)
                with c_save:
                    if st.button("💾 Save Profile", use_container_width=True):
                        if not prof_name.strip():
                            st.warning("Enter a profile name first.")
                        elif len(profiles) >= 3 and prof_name.strip() not in profiles:
                            st.warning("Maximum 3 custom profiles allowed. Overwrite an existing name or clear.")
                        else:
                            profiles[prof_name.strip()] = {
                                "base_url": st.session_state["llm_base_url"],
                                "model": st.session_state["llm_model"],
                                "api_key": st.session_state.get("llm_api_key", ""),
                                "extra_headers": st.session_state.get("llm_extra_headers_json", ""),
                            }
                            st.session_state["saved_custom_profiles"] = profiles
                            st.success(f"Saved '{prof_name.strip()}' in session memory.")
                with c_load:
                    if profiles:
                        chosen_prof = st.selectbox("Saved Profiles", options=list(profiles.keys()), label_visibility="collapsed")
                        if st.button("📂 Load", use_container_width=True):
                            p_data = profiles[chosen_prof]
                            st.session_state["llm_base_url"] = p_data["base_url"]
                            st.session_state["llm_model"] = p_data["model"]
                            st.session_state["llm_api_key"] = p_data["api_key"]
                            st.session_state["llm_extra_headers_json"] = p_data["extra_headers"]
                            st.rerun()

        else:
            # Standard Provider (OpenAI, Google Gemini, Groq)
            default_models = PROVIDER_MODELS.get(provider, ["gpt-4o-mini"])
            use_custom_name = st.checkbox(
                "Use custom model name",
                value=st.session_state.get("llm_custom_model_toggle", False),
            )
            st.session_state["llm_custom_model_toggle"] = use_custom_name
            if use_custom_name:
                st.session_state["llm_model"] = st.text_input(
                    "Custom Model ID",
                    value=st.session_state.get("llm_model", default_models[0]),
                ).strip()
            else:
                cur_m = st.session_state.get("llm_model")
                m_idx = default_models.index(cur_m) if cur_m in default_models else 0
                st.session_state["llm_model"] = st.selectbox("Model", options=default_models, index=m_idx)

        api_key_input = st.text_input(
            "API Key (optional if set in .env / secrets or local Ollama)",
            value=st.session_state.get("llm_api_key", ""),
            type="password",
            help="Stored strictly in `st.session_state` for your current browser session — never written to disk or logs.",
        )
        st.session_state["llm_api_key"] = api_key_input.strip()

        c_temp, c_tok = st.columns(2)
        with c_temp:
            st.session_state["llm_temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("llm_temperature", 0.3)),
                step=0.05,
            )
        with c_tok:
            st.session_state["llm_max_tokens"] = st.number_input(
                "Max Tokens",
                min_value=256,
                max_value=8192,
                value=int(st.session_state.get("llm_max_tokens", 3000)),
                step=256,
            )

        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner(f"Testing connection to {provider}..."):
                ok, msg = test_llm_connection(
                    provider=st.session_state["llm_provider"],
                    model=st.session_state["llm_model"],
                    api_key=st.session_state.get("llm_api_key") or None,
                    base_url=st.session_state.get("llm_base_url") or None,
                    temperature=st.session_state["llm_temperature"],
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("---")
        st.markdown("## 🎓 Researcher Profile")
        st.session_state["researcher_field"] = st.text_input(
            "Field / Domain",
            value=st.session_state.get("researcher_field", "Agricultural & Environmental AI"),
        )
        stages = [
            "BS / Undergraduate Student",
            "MS / Master's Student",
            "PhD Student / Candidate",
            "Postdoctoral Researcher",
            "Professor / Principal Investigator",
            "Industry R&D Scientist",
        ]
        cur_stage = st.session_state.get("researcher_stage", "PhD Student / Candidate")
        s_idx = stages.index(cur_stage) if cur_stage in stages else 2
        st.session_state["researcher_stage"] = st.selectbox("Career Stage", options=stages, index=s_idx)

        styles = ["APA", "IEEE", "Harvard", "MLA", "Vancouver"]
        cur_style = st.session_state.get("citation_style", "APA")
        c_idx = styles.index(cur_style) if cur_style in styles else 0
        st.session_state["citation_style"] = st.selectbox("Citation Style", options=styles, index=c_idx)

        st.session_state["target_journal"] = st.text_input(
            "Target Journal / Conference",
            value=st.session_state.get("target_journal", "Nature / IEEE / Elsevier"),
        )

        st.markdown("---")
        ref_count = len(st.session_state.get("verified_references", []))
        st.metric("📚 Verified References in Library", ref_count)
        st.caption("Populated strictly from real API searches & uploaded PDFs.")

    return st.session_state["active_page"]
