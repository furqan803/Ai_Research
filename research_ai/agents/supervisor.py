"""
agents/supervisor.py
====================
Agent 5: AI PhD Supervisor Agent
Tagline: "Your always-available research mentor."

Capabilities:
- Creates a personalized 5-Stage Research Roadmap with milestones:
  1. Plan   -> Define RQ, supervisor expectations, literature matrix
  2. Guide  -> Methodology, ethics approval, data collection/pipeline
  3. Feedback -> Interim analysis, critical assumption checks, seminar feedback
  4. Improve -> Manuscript drafting, peer-review revisions, replication package
  5. Succeed -> Thesis submission, viva/defense mastery, societal impact
- Challenges assumptions with friendly, constructive Socratic critical questions (Critical Review Mode)
- Helps with conference presentations, academic posters, and PhD defense preparation
  (12-slide presentation outline, 4-quadrant academic poster layout, and 10 Mock Viva Questions with model answers)
- Teaches & audits Research Ethics (CRediT authorship, plagiarism prevention, AI-use disclosure,
  data integrity, conflict of interest, copyright & open access)
- Provides 24/7 conversational mentorship with conversation memory
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm_factory import invoke_llm_text, stream_llm_text


RESEARCH_ETHICS_MODULES: List[Dict[str, str]] = [
    {
        "Topic": "1. Authorship & CRediT Taxonomy",
        "Core Standard": "ICMJE & CRediT (14 contributor roles: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Supervision, Project Administration, Funding Acquisition).",
        "Actionable Rule": "Discuss and document authorship order early. Every listed author must make a substantial intellectual contribution and approve the final manuscript. AI tools CANNOT be listed as co-authors.",
    },
    {
        "Topic": "2. Responsible AI-Use Disclosure",
        "Core Standard": "COPE, Nature, Science, IEEE, Elsevier & Springer Nature AI Policies.",
        "Actionable Rule": "Always include an explicit 'Declaration of Generative AI and AI-Assisted Technologies in the Writing Process' section stating which AI tool/model was used and confirming that human authors verified all outputs, citations, and code.",
    },
    {
        "Topic": "3. Plagiarism & Self-Plagiarism Prevention",
        "Core Standard": "Committee on Publication Ethics (COPE) Originality Guidelines.",
        "Actionable Rule": "Never copy sentences verbatim without quotation marks and citation. Paraphrase ideas in your own scholarly voice, cite primary sources (not secondary summaries alone), and avoid recycling text across your own papers ('text recycling').",
    },
    {
        "Topic": "4. Data Integrity, Reproducibility & Pre-Registration",
        "Core Standard": "FAIR Data Principles (Findable, Accessible, Interoperable, Reusable).",
        "Actionable Rule": "Never cherry-pick p-values (p-hacking) or hypothesize after results are known (HARKing). Archive raw/processed data and analysis scripts on Zenodo, Figshare, or GitHub with a permanent DOI.",
    },
    {
        "Topic": "5. Conflict of Interest (COI) & Human/Animal Ethics (IRB/IACUC)",
        "Core Standard": "Declaration of Helsinki & Institutional Review Board (IRB) Compliance.",
        "Actionable Rule": "Obtain IRB/ethics approval BEFORE collecting human or sensitive data. Declare all financial, institutional, or personal relationships that could be perceived as influencing the research.",
    },
    {
        "Topic": "6. Copyright, Licensing & Open Access",
        "Core Standard": "Creative Commons (CC BY 4.0) & Sherpa Romeo Archiving Policies.",
        "Actionable Rule": "Check whether your funder requires Green or Gold Open Access. Retain rights to deposit your Author Accepted Manuscript (AAM) in your institutional repository or preprint server (arXiv/bioRxiv/medRxiv).",
    },
]


def generate_phd_roadmap_and_critique(
    llm: BaseChatModel,
    research_topic: str,
    field: str,
    career_stage: str = "PhD Student",
    program_duration: str = "3-Year PhD Journey (36 Months)",
    current_draft_or_assumptions: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate:
    1. Personalized 5-Stage Roadmap (Plan -> Guide -> Feedback -> Improve -> Succeed)
    2. Friendly Critical Review Mode (challenging assumptions & strengthening rigor)
    """
    system_prompt = (
        "You are the **AI PhD Supervisor Agent** inside ResearchAI ('Your always-available research mentor.').\n"
        "Your tone is warm, encouraging, intellectually rigorous, and deeply supportive of students and researchers "
        "from BS level to PhD and postdoctoral stages."
    )

    roadmap_prompt = f"""Create a personalized, milestone-driven **Research Roadmap** organized around the 5 ResearchAI Mentorship Pillars (**Plan -> Guide -> Feedback -> Improve -> Succeed**):

- **Research Topic**: {research_topic}
- **Field / Discipline**: {field}
- **Researcher Career Stage**: {career_stage}
- **Timeline Horizon**: {program_duration}

Include:
1. **Pillar 1 — PLAN (Months 1–6)**: Problem formulation, supervisor charter, literature mastery, gap validation
2. **Pillar 2 — GUIDE (Months 7–14)**: Experimental design, ethics/IRB clearance, baseline pipeline & pilot study
3. **Pillar 3 — FEEDBACK (Months 15–22)**: Main experiments, statistical validation, conference presentation, mid-term qualifying review
4. **Pillar 4 — IMPROVE (Months 23–30)**: Journal manuscript submission, handling peer-review revisions, open-science replication package
5. **Pillar 5 — SUCCEED (Months 31–36)**: Thesis integration, mock viva / oral defense, career transition & societal impact
6. **Monthly / Quarterly Habit & Wellbeing Advice** for sustainable research productivity.
"""

    roadmap_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=roadmap_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    critique_prompt = f"""Act in **Friendly Critical Review Mode** as a supportive PhD Supervisor reviewing the student's current research plan:
- **Topic**: {research_topic}
- **Student's Stated Assumptions / Draft Plan**: {current_draft_or_assumptions or 'Standard assumptions for this topic in ' + field}

Provide:
1. **5 Constructive Socratic Questions** that challenge hidden assumptions (data quality, confounding variables, baseline comparisons, scalability, real-world deployment).
2. **How to Answer Each Question**: Practical experiments, ablation studies, or literature checks the student can run to make their thesis bulletproof.
"""
    critique_md = invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=critique_prompt,
        provider=provider_name,
        base_url=base_url,
    )

    return {
        "roadmap_md": roadmap_md,
        "critique_md": critique_md,
    }


def generate_defense_and_presentation_kit(
    llm: BaseChatModel,
    research_topic: str,
    field: str,
    manuscript_or_summary: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> str:
    """
    Generate a complete Presentation, Academic Poster & Viva/Defense Preparation Kit:
    - 12-Slide Defense/Conference Presentation Outline
    - 4-Section Academic Poster Layout
    - 10 Tough Viva / Defense Questions with Model High-Scoring Answers
    """
    system_prompt = (
        "You are an experienced PhD Thesis Examiner and Academic Communication Coach. "
        "Help the researcher prepare an outstanding presentation, poster, and oral defense (viva)."
    )

    user_prompt = f"""Prepare a comprehensive **Presentation, Poster & Viva Defense Kit** for:
- **Research Topic**: {research_topic}
- **Field**: {field}
- **Context / Summary**: {manuscript_or_summary[:3000] if manuscript_or_summary else 'Full research lifecycle project'}

Structure your output in Markdown with:
## 1. 12-Slide Conference / Thesis Defense Presentation Outline
- For each slide (Slide 1 to Slide 12): **Slide Title**, **Visual/Figure to Show**, and **3 Key Spoken Talking Points**.

## 2. Academic Conference Poster Layout (4-Quadrant Design)
- **Header Banner**, **Column 1 (Motivation & Gap)**, **Column 2 (Methodology & Architecture Diagram)**, **Column 3 (Key Empirical Figures & Tables)**, **Column 4 (Conclusions, QR Code to Paper/Code & References)**.

## 3. Ten (10) Mock PhD Viva / Oral Defense Questions & Model Answers
- Cover: Originality/Contribution, Why this Method over Alternatives, Statistical Power & Limitations, Ethical/AI Disclosure, and 5-Year Future Vision. Include a strong **Model Answer Strategy** for each question.
"""
    return invoke_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider_name,
        base_url=base_url,
    )


def stream_supervisor_chat(
    llm: BaseChatModel,
    user_message: str,
    chat_history: List[Dict[str, str]],
    researcher_profile: Dict[str, str],
    project_context: str = "",
    provider_name: str = "",
    base_url: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream 24/7 academic mentorship responses with conversation memory and awareness
    of the researcher's profile and project outputs.
    """
    system_prompt = (
        "You are the **AI PhD Supervisor Agent** inside ResearchAI ('Your always-available research mentor.').\n"
        f"Researcher Profile: Career Stage = {researcher_profile.get('stage', 'PhD Student')}, "
        f"Field = {researcher_profile.get('field', 'Research')}, "
        f"Target Venue = {researcher_profile.get('target_journal', 'Peer-reviewed journal')}.\n"
        f"Current Project Context:\n{project_context[:2000]}\n\n"
        "Provide clear, empathetic, structured, and actionable academic mentorship. "
        "Remind the researcher to verify empirical claims and consult their human institutional supervisor on formal policy matters."
    )
    return stream_llm_text(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_message,
        chat_history=chat_history,
        provider=provider_name,
        base_url=base_url,
    )
