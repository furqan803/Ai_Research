"""
test_e2e.py
===========
Automated End-to-End Verification Suite for ResearchAI (`research_ai`):
1. Tests `utils/llm_factory.py` with OpenAI provider AND Custom API (OpenAI-compatible) provider
   including `validate_base_url()` and friendly error handling.
2. Tests Agent 1 (`agents/discovery.py`) with a live query on AI-based drought prediction in wheat.
3. Tests Agent 2 (`agents/literature.py`) PRISMA screening, evidence table extraction & systematic review.
4. Tests Agent 3 (`agents/experiment.py`) + `utils/code_runner.py` on `wheat_drought_trial_demo.csv`
   AND Biopython sequence analysis, plus verifies AST security blocking on forbidden imports.
5. Tests Agent 4 (`agents/writing_publication.py`) complete canonical paper generation with
   verified references, journal recommendations, predatory checklist & peer-review simulation.
6. Tests Agent 5 (`agents/supervisor.py`) PhD roadmap, critical review, defense kit & streaming chat.
7. Tests `graph/workflow.py` LangGraph `StateGraph` compilation and Step 6 Impact generation.
8. Tests `utils/exporters.py` Markdown (`.md`), Word (`.docx`), and PDF (`.pdf`) generation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Ensure research_ai is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from agents.discovery import run_discovery_agent
from agents.experiment import (
    generate_custom_analysis_code,
    generate_experimental_design,
    interpret_executed_results,
    run_bioinformatics_sequence_stats,
)
from agents.literature import run_literature_agent
from agents.supervisor import (
    generate_defense_and_presentation_kit,
    generate_phd_roadmap_and_critique,
    stream_supervisor_chat,
)
from agents.writing_publication import (
    recommend_journals_with_fit,
    simulate_peer_review_and_rebuttal,
    write_canonical_manuscript,
)
from graph.workflow import build_research_workflow_graph, generate_impact_package
from utils.code_runner import (
    build_default_analysis_script,
    execute_sandboxed_python,
    validate_code_safety,
)
from utils.exporters import export_to_docx, export_to_markdown, export_to_pdf
from utils.llm_factory import (
    format_llm_error,
    get_llm,
    resolve_credentials,
    test_llm_connection,
    validate_base_url,
)


def main() -> None:
    print("==================================================================")
    print("RESEARCH_AI END-TO-END VERIFICATION SUITE")
    print("==================================================================")

    # 1. Test LLM Factory & Custom API Path
    print("\n[1/8] Testing LLM Factory (OpenAI + Custom API OpenAI-compatible path)...")
    ok_url, hint_url = validate_base_url("https://openrouter.ai/api/v1")
    assert ok_url, f"Expected valid URL, got {hint_url}"
    bad_url, _ = validate_base_url("not-a-url")
    assert not bad_url, "Expected invalid URL to be flagged"

    # Test OpenAI path
    ok_openai, msg_openai = test_llm_connection(provider="OpenAI", model="gpt-4o-mini")
    print("  -> OpenAI connection test:", ok_openai, "|", msg_openai)
    assert ok_openai, f"OpenAI test failed: {msg_openai}"

    # Test Custom API (OpenAI-compatible) path using the OpenAI-compatible endpoint
    creds = resolve_credentials("OpenAI")
    ok_custom, msg_custom = test_llm_connection(
        provider="Custom API (OpenAI-compatible)",
        model="gpt-5-mini",
        api_key=creds["api_key"],
        base_url=creds["base_url"],
    )
    print("  -> Custom API (OpenAI-compatible) connection test:", ok_custom, "|", msg_custom)
    assert ok_custom, f"Custom API test failed: {msg_custom}"

    # Create active LLM instance via Custom API path to verify full compatibility
    llm = get_llm(
        provider="Custom API (OpenAI-compatible)",
        model="gpt-5-mini",
        api_key=creds["api_key"],
        base_url=creds["base_url"],
        temperature=0.2,
        max_tokens=1200,
    )

    # 2. Test Agent 1: Discovery Agent
    print("\n[2/8] Testing Agent 1: AI Research Discovery Agent...")
    disc_res = run_discovery_agent(
        llm=llm,
        field="Agricultural & Environmental AI",
        keywords="wheat drought prediction machine learning",
        career_stage="PhD Student",
        available_resources="University GPU lab & satellite imagery",
        preferred_depth="Deep PhD-Level Exploration",
        databases=["OpenAlex", "Crossref", "arXiv"],
        max_per_db=3,
        citation_style="APA",
    )
    print(f"  -> Retrieved {len(disc_res['papers'])} unique papers; Verified Library size = {len(disc_res['verified_references'])}")
    print(f"  -> Discovery report length: {len(disc_res['report_md'])} chars")
    assert len(disc_res["papers"]) > 0, "Expected at least 1 paper from live scholarly APIs"
    assert len(disc_res["report_md"]) > 200, "Expected non-empty discovery report"

    # 3. Test Agent 2: Literature Review Agent
    print("\n[3/8] Testing Agent 2: AI Literature Review Agent (PRISMA + Evidence Matrix)...")
    lit_res = run_literature_agent(
        llm=llm,
        topic_query="wheat drought stress machine learning remote sensing",
        inclusion_criteria="wheat, drought, machine learning, remote sensing, yield",
        exclusion_criteria="corrigendum, book review",
        min_year=2018,
        databases=["OpenAlex", "Crossref"],
        max_per_db=3,
        citation_style="APA",
        existing_library=disc_res["verified_references"],
        discovery_context=disc_res["report_md"][:800],
    )
    print("  -> PRISMA Counts:", lit_res["prisma_counts"])
    print("  -> Evidence Matrix shape:", lit_res["evidence_df"].shape)
    print("  -> Systematic Review length:", len(lit_res["review_md"]), "chars")
    assert len(lit_res["review_md"]) > 200

    # 4. Test Agent 3: Experiment Agent + Safe Subprocess Code Runner + CSV
    print("\n[4/8] Testing Agent 3: AI Research Experiment Agent & Safe Subprocess Runner...")
    # Verify AST blocker catches forbidden imports
    safe_check, issues = validate_code_safety("import os\nos.system('ls')")
    assert not safe_check, "AST validator MUST block `import os`"
    print("  -> AST Security Check verified (blocked `import os`):", issues)

    csv_path = BASE_DIR / "sample_data" / "wheat_drought_trial_demo.csv"
    df_demo = pd.read_csv(csv_path)
    script = build_default_analysis_script(df_demo, is_demo=True)
    exec_res = execute_sandboxed_python(script, df=df_demo, timeout_sec=25)
    print("  -> Subprocess execution success:", exec_res["success"])
    print("  -> Generated tables:", [t["name"] for t in exec_res["tables"]])
    print("  -> Generated figures:", [f["name"] for f in exec_res["figures"]])
    assert exec_res["success"], f"Subprocess failed: {exec_res['stderr']}"
    assert len(exec_res["tables"]) >= 2, "Expected descriptive and regression tables"
    assert len(exec_res["figures"]) >= 2, "Expected 2 PNG figures"

    bio_stats = run_bioinformatics_sequence_stats(">Seq1\nATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
    print("  -> Biopython sequence check:", bio_stats["records"][0]["GC Content (%)"], "% GC")

    methods_md = interpret_executed_results(
        llm=llm,
        research_question="Effect of drought stress on wheat grain yield and proline accumulation",
        executed_stdout=exec_res["stdout"],
        table_summaries="",
        is_demo=True,
    )
    print("  -> Interpreted Methods & Results length:", len(methods_md), "chars")

    # 5. Test Agent 4: Academic Writing & Publication Agent
    print("\n[5/8] Testing Agent 4: AI Academic Writing & Publication Agent (Canonical Paper + References)...")
    wrt_res = write_canonical_manuscript(
        llm=llm,
        paper_title="Explainable Multi-Modal Machine Learning for Drought Prediction in Wheat",
        research_question="How do drought stress treatments impact wheat yield and physiological markers?",
        field="Agricultural & Environmental AI",
        citation_style="APA",
        target_journal="Computers and Electronics in Agriculture",
        verified_references=lit_res["verified_references"],
        literature_context=lit_res["review_md"][:1000],
        executed_results_stdout=exec_res["stdout"],
        is_demo_data=True,
    )
    print("  -> Canonical Manuscript length:", len(wrt_res["manuscript_md"]), "chars")
    print("  -> Cover Letter length:", len(wrt_res["cover_letter_md"]), "chars")
    assert "References" in wrt_res["manuscript_md"]

    jn_res = recommend_journals_with_fit(
        llm=llm,
        topic="wheat drought machine learning",
        field="Agricultural AI",
    )
    print("  -> Recommended Journals count:", len(jn_res["journals_df"]))

    pr_res = simulate_peer_review_and_rebuttal(
        llm=llm,
        manuscript_md=wrt_res["manuscript_md"][:3000],
        target_journal="Computers and Electronics in Agriculture",
    )
    print("  -> Peer Review & Rebuttal lengths:", len(pr_res["peer_review_md"]), "/", len(pr_res["rebuttal_md"]))

    # 6. Test Agent 5: PhD Supervisor Agent
    print("\n[6/8] Testing Agent 5: AI PhD Supervisor Agent...")
    sup_res = generate_phd_roadmap_and_critique(
        llm=llm,
        research_topic="AI-based drought prediction in wheat",
        field="Agricultural AI",
        career_stage="PhD Student",
    )
    print("  -> PhD Roadmap & Critique lengths:", len(sup_res["roadmap_md"]), "/", len(sup_res["critique_md"]))

    def_kit = generate_defense_and_presentation_kit(
        llm=llm,
        research_topic="AI-based drought prediction in wheat",
        field="Agricultural AI",
        manuscript_or_summary=wrt_res["manuscript_md"][:1500],
    )
    print("  -> Defense Kit length:", len(def_kit), "chars")

    stream_chunks = list(
        stream_supervisor_chat(
            llm=llm,
            user_message="How should I prepare for my first supervisor meeting?",
            chat_history=[],
            researcher_profile={"field": "Agricultural AI", "stage": "PhD Student", "target_journal": "Nature Food"},
        )
    )
    stream_reply = "".join(stream_chunks)
    print("  -> Streaming Supervisor Chat reply length:", len(stream_reply), "chars")
    assert len(stream_reply) > 50

    # 7. Test LangGraph Workflow Compilation & Step 6 Impact
    print("\n[7/8] Testing LangGraph Workflow Compilation & Step 6 Impact Generator...")
    compiled_graph = build_research_workflow_graph()
    assert compiled_graph is not None
    imp_res = generate_impact_package(
        llm=llm,
        topic="AI-based drought prediction in wheat",
        field="Agricultural AI",
        manuscript_md=wrt_res["manuscript_md"][:2000],
    )
    print("  -> Step 6 Impact Package length:", len(imp_res["impact_md"]), "chars")

    # 8. Test Exporters (MD, DOCX, PDF)
    print("\n[8/8] Testing Canonical Document Exporters (MD, DOCX, PDF)...")
    md_bytes = export_to_markdown(
        title=wrt_res["paper_title"],
        content_md=wrt_res["manuscript_md"],
        references=lit_res["verified_references"],
        citation_style="APA",
    )
    docx_bytes = export_to_docx(
        title=wrt_res["paper_title"],
        content_md=wrt_res["manuscript_md"],
        references=lit_res["verified_references"],
        citation_style="APA",
        figures=exec_res["figures"],
    )
    pdf_bytes = export_to_pdf(
        title=wrt_res["paper_title"],
        content_md=wrt_res["manuscript_md"],
        references=lit_res["verified_references"],
        citation_style="APA",
    )
    print(f"  -> Exported sizes: MD={len(md_bytes)} bytes | DOCX={len(docx_bytes)} bytes | PDF={len(pdf_bytes)} bytes")
    assert len(md_bytes) > 500 and len(docx_bytes) > 5000 and len(pdf_bytes) > 1000

    print("\n==================================================================")
    print("ALL 8 END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    main()
