"""
graph/workflow.py
=================
LangGraph Multi-Agent Orchestrator for ResearchAI.

Implements the complete 6-Step Research Pipeline as a stateful `StateGraph`:
  1. Discover Ideas       (`discovery_node`)
  2. Review Literature    (`literature_node`)
  3. Design & Analyze     (`experiment_node`)
  4. Write & Publish      (`writing_node`)
  5. Get Mentored         (`supervisor_node`)
  6. Create Impact        (`impact_node`)

Supports:
- Human-in-the-loop approval checkpoints between every step (`run_workflow_step`)
- Full pipeline execution with shared `ResearchWorkflowState`
- Step 6 "Create Impact" generator (Plain-Language Summary, LinkedIn/Social Post,
  Policy/Industry Brief, Press-Release Summary, Visibility Plan [ORCID, Google Scholar,
  Preprints, Conferences], and Citation/Impact Tracking Checklist)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agents.discovery import run_discovery_agent
from agents.experiment import (
    generate_custom_analysis_code,
    generate_experimental_design,
    interpret_executed_results,
)
from agents.literature import run_literature_agent
from agents.supervisor import generate_phd_roadmap_and_critique
from agents.writing_publication import (
    recommend_journals_with_fit,
    write_canonical_manuscript,
)
from utils.code_runner import build_default_analysis_script, execute_sandboxed_python
from utils.llm_factory import get_llm, invoke_llm_text


class ResearchWorkflowState(TypedDict, total=False):
    # Configuration & Inputs
    provider: str
    model: str
    api_key: Optional[str]
    base_url: Optional[str]
    temperature: float
    max_tokens: int
    field: str
    topic: str
    career_stage: str
    available_resources: str
    citation_style: str
    target_journal: str
    stop_after_step: int  # Human-approval checkpoint control (1..6)

    # Shared Artifacts Across Nodes
    verified_references: List[Dict[str, Any]]
    discovery_output: Dict[str, Any]
    literature_output: Dict[str, Any]
    experiment_output: Dict[str, Any]
    writing_output: Dict[str, Any]
    supervisor_output: Dict[str, Any]
    impact_output: Dict[str, Any]
    completed_steps: List[int]


def generate_impact_package(
    llm: BaseChatModel,
    topic: str,
    field: str,
    manuscript_md: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Step 6: Create Impact — Produces:
    1. Plain-Language Summary (for the general public & non-specialists)
    2. LinkedIn / X (Twitter) Academic Announcement Posts
    3. Policy & Industry Executive Brief
    4. University Press-Release Style Summary
    5. Scholarly Visibility Plan (ORCID, Google Scholar, arXiv/bioRxiv Preprints, Conferences)
    6. Citation & Research Impact Tracking Checklist
    """
    system_prompt = (
        "You are the **Research Impact & Science Communication Strategist** inside ResearchAI "
        "(Step 6: 'Create Impact — Because better research creates a better world.').\n"
        "Translate rigorous academic findings into accurate, engaging outreach materials without "
        "exaggerating claims beyond the empirical evidence."
    )

    user_prompt = f"""Create a complete **Step 6: Research Impact & Outreach Package** for the following project:
- **Research Topic**: {topic}
- **Field / Discipline**: {field}
- **Manuscript / Project Context**:
{manuscript_md[:4500] if manuscript_md else 'Research project on ' + topic}

Please structure the Impact Package in Markdown with all 6 required deliverables:

## 1. Plain-Language Summary (Lay Abstract)
- A clear, jargon-free 200-word explanation accessible to high-school students, citizens, and non-specialist stakeholders.

## 2. Social Media & Academic Network Posts (LinkedIn & X/Bluesky)
- **LinkedIn Long-Form Post** (structured with hook, key problem, methodology, key takeaway, call for collaboration, and relevant hashtags)
- **Short 5-Post Thread** for X / Bluesky / Mastodon announcing the paper and code repository.

## 3. Policy & Industry R&D Executive Brief
- **Key Issue**, **Evidence-Based Findings**, and **3 Actionable Recommendations** for policymakers, regulators, or industry R&D leaders.

## 4. University Press-Release Style Summary
- Headline, Dateline, Lead Paragraph, Quoted Researcher Perspective placeholder, and Societal Significance.

## 5. Academic Visibility & Open-Science Dissemination Plan
- Concrete step-by-step action plan covering **ORCID**, **Google Scholar Profile**, **Preprint Servers (arXiv / bioRxiv / SSRN)**, **Zenodo / GitHub Code & Data DOI archiving**, and **Target Academic Conferences**.

## 6. Citation & Impact Tracking Checklist
- Checklist for monitoring citations, OpenAlex/Semantic Scholar alerts, Altmetric attention, policy mentions, and reproducibility requests.
"""

    impact_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )
    return {
        "topic": topic,
        "field": field,
        "impact_md": impact_md,
    }


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def _build_node_llm(state: ResearchWorkflowState) -> BaseChatModel:
    return get_llm(
        provider=state.get("provider", "OpenAI"),
        model=state.get("model", "gpt-4o-mini"),
        api_key=state.get("api_key"),
        base_url=state.get("base_url"),
        temperature=float(state.get("temperature", 0.3)),
        max_tokens=int(state.get("max_tokens", 2800)),
    )


def discovery_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    res = run_discovery_agent(
        llm=llm,
        field=state.get("field", "General Science"),
        keywords=state.get("topic", "AI in scientific discovery"),
        career_stage=state.get("career_stage", "PhD Student"),
        available_resources=state.get("available_resources", "University compute & open datasets"),
        citation_style=state.get("citation_style", "APA"),
        existing_library=state.get("verified_references", []),
        max_per_db=4,
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    done = list(state.get("completed_steps", []))
    if 1 not in done:
        done.append(1)
    return {
        "discovery_output": res,
        "verified_references": res["verified_references"],
        "completed_steps": done,
    }


def literature_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    disc_md = (state.get("discovery_output") or {}).get("report_md", "")
    res = run_literature_agent(
        llm=llm,
        topic_query=f"{state.get('field', '')} {state.get('topic', '')}".strip(),
        citation_style=state.get("citation_style", "APA"),
        existing_library=state.get("verified_references", []),
        discovery_context=disc_md,
        max_per_db=4,
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    done = list(state.get("completed_steps", []))
    if 2 not in done:
        done.append(2)
    return {
        "literature_output": res,
        "verified_references": res["verified_references"],
        "completed_steps": done,
    }


def experiment_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    lit_md = (state.get("literature_output") or {}).get("review_md", "")
    demo_csv = Path(__file__).resolve().parent.parent / "sample_data" / "wheat_drought_trial_demo.csv"
    df_demo = pd.read_csv(demo_csv) if demo_csv.exists() else pd.DataFrame({"x": [1, 2, 3, 4], "y": [2.1, 3.9, 6.2, 8.0]})

    design_md = generate_experimental_design(
        llm=llm,
        research_question=state.get("topic", ""),
        field=state.get("field", ""),
        available_resources=state.get("available_resources", ""),
        dataset_summary=f"DEMO DATA – not real ({df_demo.shape[0]} rows, columns: {list(df_demo.columns)})",
        literature_context=lit_md,
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    script_code = build_default_analysis_script(df_demo, is_demo=True)
    exec_res = execute_sandboxed_python(script_code, df=df_demo, timeout_sec=25)
    methods_results_md = interpret_executed_results(
        llm=llm,
        research_question=state.get("topic", ""),
        executed_stdout=exec_res.get("stdout", ""),
        table_summaries="",
        is_demo=True,
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    exp_out = {
        "design_md": design_md,
        "code": script_code,
        "execution": exec_res,
        "methods_results_md": methods_results_md,
        "is_demo": True,
    }
    done = list(state.get("completed_steps", []))
    if 3 not in done:
        done.append(3)
    return {
        "experiment_output": exp_out,
        "completed_steps": done,
    }


def writing_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    lit_md = (state.get("literature_output") or {}).get("review_md", "")
    exp_out = state.get("experiment_output") or {}
    exec_stdout = ((exp_out.get("execution") or {}).get("stdout")) or ""

    paper_res = write_canonical_manuscript(
        llm=llm,
        paper_title=f"A Multi-Method Investigation of {state.get('topic', 'Scientific Inquiry').title()}",
        research_question=state.get("topic", ""),
        field=state.get("field", ""),
        citation_style=state.get("citation_style", "APA"),
        target_journal=state.get("target_journal", "Peer-Reviewed Journal"),
        verified_references=state.get("verified_references", []),
        literature_context=lit_md,
        experiment_design_context=exp_out.get("design_md", ""),
        executed_results_stdout=exec_stdout,
        is_demo_data=bool(exp_out.get("is_demo", True)),
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    journal_res = recommend_journals_with_fit(
        llm=llm,
        topic=state.get("topic", ""),
        field=state.get("field", ""),
        target_journal_hint=state.get("target_journal", ""),
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    paper_res.update(journal_res)
    done = list(state.get("completed_steps", []))
    if 4 not in done:
        done.append(4)
    return {
        "writing_output": paper_res,
        "completed_steps": done,
    }


def supervisor_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    sup_res = generate_phd_roadmap_and_critique(
        llm=llm,
        research_topic=state.get("topic", ""),
        field=state.get("field", ""),
        career_stage=state.get("career_stage", "PhD Student"),
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    done = list(state.get("completed_steps", []))
    if 5 not in done:
        done.append(5)
    return {
        "supervisor_output": sup_res,
        "completed_steps": done,
    }


def impact_node(state: ResearchWorkflowState) -> Dict[str, Any]:
    llm = _build_node_llm(state)
    manuscript_md = (state.get("writing_output") or {}).get("manuscript_md", "")
    imp_res = generate_impact_package(
        llm=llm,
        topic=state.get("topic", ""),
        field=state.get("field", ""),
        manuscript_md=manuscript_md,
        provider_name=state.get("provider", ""),
        base_url=state.get("base_url"),
    )
    done = list(state.get("completed_steps", []))
    if 6 not in done:
        done.append(6)
    return {
        "impact_output": imp_res,
        "completed_steps": done,
    }


# ---------------------------------------------------------------------------
# Conditional Edges for Human-Approval Checkpoints
# ---------------------------------------------------------------------------

def _make_gate(step_num: int, next_node_name: str):
    def _gate(state: ResearchWorkflowState) -> str:
        stop_at = int(state.get("stop_after_step", 6))
        if stop_at <= step_num:
            return END
        return next_node_name
    return _gate


def build_research_workflow_graph():
    """
    Build and compile the 6-node LangGraph `StateGraph` with human-approval checkpoint routing.
    """
    builder = StateGraph(ResearchWorkflowState)

    builder.add_node("discover", discovery_node)
    builder.add_node("literature", literature_node)
    builder.add_node("experiment", experiment_node)
    builder.add_node("writing", writing_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("impact", impact_node)

    builder.add_edge(START, "discover")
    builder.add_conditional_edges("discover", _make_gate(1, "literature"))
    builder.add_conditional_edges("literature", _make_gate(2, "experiment"))
    builder.add_conditional_edges("experiment", _make_gate(3, "writing"))
    builder.add_conditional_edges("writing", _make_gate(4, "supervisor"))
    builder.add_conditional_edges("supervisor", _make_gate(5, "impact"))
    builder.add_edge("impact", END)

    return builder.compile()


def run_single_workflow_stage(step_number: int, state: ResearchWorkflowState) -> Dict[str, Any]:
    """
    Execute a single stage (1..6) of the LangGraph workflow so the Streamlit UI can
    pause at a Human-Approval Checkpoint between every step.
    """
    dispatch = {
        1: discovery_node,
        2: literature_node,
        3: experiment_node,
        4: writing_node,
        5: supervisor_node,
        6: impact_node,
    }
    fn = dispatch.get(step_number)
    if not fn:
        raise ValueError(f"Invalid workflow step: {step_number}")
    return fn(state)
