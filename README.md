# 🔬 ResearchAI (`research_ai`)

> **“Five AI Agents. One Smarter Research Journey.”**  
> *From idea to impact — your AI research team.*  
> **Motto:** *“Higher Research, Brighter Tomorrow.”*

**ResearchAI** is a complete, production-ready multi-agent scientific research web application built with **Python, Streamlit, LangGraph, and LangChain**. It empowers **BS students, MS/PhD candidates, researchers, professors, universities, research institutes, and industrial R&D teams** to go from an initial curiosity to a **canonically written, reference-verified academic paper** and global impact package.

---

## 🌟 Core Architecture & The 5 Specialized AI Agents

| # | Agent | Tagline | Key Capabilities |
| :- | :--- | :--- | :--- |
| **1** | 🔵 **AI Research Discovery Agent** (`agents/discovery.py`) | *“Find what matters. Discover new possibilities.”* | Scans **arXiv, Semantic Scholar, OpenAlex, Crossref, and PubMed**; plots publication trends per year, top authors & venues; identifies evidence-backed gaps; generates **5 novel research ideas** with testable RQs, novelty checks, feasibility & impact. |
| **2** | 🟢 **AI Literature Review Agent** (`agents/literature.py`) | *“Turn thousands of papers into clear insights.”* | Runs a transparent **PRISMA-style screening flow** (`Identified → Deduplicated → Screened → Included`); documents inclusion/exclusion reasons; extracts comparison tables; reads uploaded PDFs (`pypdf`); builds a **Verified Reference Library** with real DOIs/links. |
| **3** | 🟠 **AI Research Experiment Agent** (`agents/experiment.py`) | *“Design. Analyze. Discover.”* | Designs hypotheses ($H_0/H_1$), variables, controls & power reasoning; executes real statistical (`scipy`, `statsmodels`), ML (`scikit-learn`), and bioinformatics (`biopython`) code on CSV/Excel data inside a **restricted sandboxed subprocess** after human code review. **Zero fabricated numbers.** |
| **4** | 🟣 **AI Academic Writing & Publication Agent** (`agents/writing_publication.py`) | *“From data to publication.”* | Writes complete, canonically formatted manuscripts (`Abstract` to `References`) using **only verified references** and **actual executed results** in **APA, IEEE, Harvard, MLA, or Vancouver** style; recommends journals via OpenAlex/Crossref; simulates **2-Reviewer + Editor peer review**; writes rebuttal & cover letters; exports **DOCX, PDF & Markdown**. |
| **5** | 🩵 **AI PhD Supervisor Agent** (`agents/supervisor.py`) | *“Your always-available research mentor.”* | Generates a 5-pillar roadmap (`Plan, Guide, Feedback, Improve, Succeed`); challenges assumptions in **Friendly Critical Review Mode**; creates **12-Slide Defense Outlines, Poster Layouts & 10 Mock Viva Q&As**; teaches research ethics (CRediT, AI disclosure, integrity) + **24/7 streaming mentor chat**. |
| **6** | 🚀 **LangGraph Full Pipeline & Impact** (`graph/workflow.py`) | *“Because better research creates a better world.”* | Orchestrates all 5 agents via a stateful LangGraph `StateGraph` with **Human-Approval Checkpoints** and produces **Step 6 Impact**: Lay Summary, LinkedIn Post, Policy Brief, Press Release, ORCID/Scholar Visibility Plan & Citation Checklist. |

---

## 🔌 Multi-Provider LLM Support (with Strong Custom API Support)

All agents route exclusively through `utils/llm_factory.py` (`get_llm(...)`):

1. **OpenAI** (`langchain-openai` / `ChatOpenAI`) — `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`
2. **Google Gemini** (`langchain-google-genai` / `ChatGoogleGenerativeAI`) — `gemini-2.0-flash`, `gemini-1.5-pro` *(Free tier available)*
3. **Groq** (`langchain-groq` / `ChatGroq`) — `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` *(Free tier available)*
4. **Custom API (OpenAI-compatible)** (`ChatOpenAI` with custom `base_url`) — One-click presets for **OpenRouter** (including `:free` models), **Together AI**, **DeepSeek**, **Ollama** (`http://localhost:11434/v1`), **LM Studio** (`http://localhost:1234/v1`), and **Other** (Mistral, xAI, Azure OpenAI, vLLM). Supports saving up to **3 named custom profiles** in `st.session_state`.

**API Key Priority:** `Sidebar Input` → `st.secrets` (`.streamlit/secrets.toml`) → `.env` → `OS Environment Variables`.

---

## 📁 Project Structure

```text
research_ai/
├── app.py                        # Main Streamlit multi-page application
├── agents/
│   ├── discovery.py              # Agent 1: AI Research Discovery Agent
│   ├── literature.py             # Agent 2: AI Literature Review Agent (PRISMA + PDF)
│   ├── experiment.py             # Agent 3: AI Research Experiment Agent (Stats/ML/Bio)
│   ├── writing_publication.py    # Agent 4: AI Academic Writing & Publication Agent
│   └── supervisor.py             # Agent 5: AI PhD Supervisor Agent
├── graph/
│   └── workflow.py               # LangGraph 6-step orchestrator + Step 6 Impact
├── utils/
│   ├── llm_factory.py            # get_llm() multi-provider factory & error handler
│   ├── search_apis.py            # Live arXiv, OpenAlex, Semantic Scholar, Crossref, PubMed & PDF clients
│   ├── reference_library.py      # Verified Reference Library & APA/IEEE/Harvard/MLA/Vancouver formatter
│   ├── code_runner.py            # AST-validated, network-isolated subprocess Python runner
│   ├── exporters.py              # DOCX (python-docx), PDF (reportlab/fpdf2) & Markdown exporters
│   └── ui.py                     # Dark-navy theme components, sidebar, stepper & download toolbar
├── sample_data/
│   └── wheat_drought_trial_demo.csv  # Labeled synthetic demo dataset ("DEMO DATA – not real")
├── assets/
│   └── style.css                 # Custom dark-navy + colour-coded agent card styling
├── .streamlit/
│   ├── config.toml               # Streamlit dark theme & server config
│   └── secrets.toml.example      # Template for Streamlit Community Cloud secrets
├── requirements.txt              # Pinned dependencies compatible with Streamlit Cloud
├── .env.example                  # Environment variable template
├── .gitignore                    # Excludes .venv, .env, secrets.toml
├── README.md                     # Project overview & quickstart
└── GUIDE.md                      # Step-by-step user manual & responsible AI guide
```

---

## 🚀 Quickstart (Local `.venv`)

```bash
cd research_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `research_ai/app.py` (or root `app.py`).
3. In **App Settings → Secrets**, paste your API keys using `.streamlit/secrets.toml.example` as a template.
4. Click **Deploy!**

---

## ⚖️ Academic Integrity & Responsible AI Use

- **Zero Fabricated Citations:** ResearchAI cites **only** real papers retrieved from scholarly APIs (arXiv, OpenAlex, Semantic Scholar, Crossref, PubMed) or uploaded PDFs, complete with DOIs/links.
- **Zero Fabricated Results:** Every statistical number, table, and figure in the Experiment and Writing agents is computed by executing real Python code on your dataset.
- **Mandatory Human Verification:** All outputs must be verified by the researcher and approved by their academic supervisor. Follow your target journal's AI disclosure policy; AI cannot be listed as a co-author.

---
*“Higher Research, Brighter Tomorrow.”*
