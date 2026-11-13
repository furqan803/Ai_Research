# 📘 ResearchAI — Complete User Guide & Responsible AI Manual

**App Title:** ResearchAI (`research_ai`)  
**Tagline:** *“Five AI Agents. One Smarter Research Journey.”*  
**Sub-tagline:** *“From idea to impact — your AI research team.”*  
**Footer Motto:** *“Higher Research, Brighter Tomorrow.”*

---

## 1. What ResearchAI Does

**ResearchAI** is a multi-agent scientific research platform built with **Python, Streamlit, LangGraph, and LangChain**. It guides BS students, MS/PhD candidates, professors, research institutes, and industrial R&D teams through the complete 6-step research lifecycle:

1. **Discover Ideas** (`agents/discovery.py`) — *“Find what matters. Discover new possibilities.”*
   - Scans **arXiv, Semantic Scholar, OpenAlex, Crossref, and PubMed (NCBI E-utilities)** in real time.
   - Maps bibliometric trends (publications per year, top authors, top venues) as interactive charts.
   - Pinpoints evidence-backed research gaps and generates **5 novel research ideas** with testable research questions, novelty checks against retrieved papers, feasibility assessments, and expected impact.

2. **Review Literature** (`agents/literature.py`) — *“Turn thousands of papers into clear insights.”*
   - Executes a transparent **PRISMA-style flow** (`Identified → Duplicates Removed → Screened → Excluded → Included`) with an interactive funnel chart.
   - Screens every paper against explicit inclusion/exclusion criteria with a documented reason per decision.
   - Extracts structured evidence tables (`Aim`, `Methods`, `Dataset/Sample`, `Key Results`, `Limitations`) and parses uploaded full-text PDFs (`pypdf`).
   - Populates a **Verified Reference Library** (only real papers returned by scholarly APIs with DOIs/links) that downstream agents must cite.

3. **Design & Analyze Experiments** (`agents/experiment.py`) — *“Design. Analyze. Discover.”*
   - Formulates testable hypotheses ($H_0 / H_1$), independent/dependent/control variables, sample-size power reasoning, protocols, and validity threat matrices.
   - Runs real statistical, machine learning, and bioinformatics (`biopython`) analyses on uploaded CSV/Excel datasets (or a clearly labeled `DEMO DATA – not real` dataset).
   - **Safe Sandboxed Code Execution**: Shows generated Python code to the researcher **before** running it, validates the AST against a strict scientific allow-list, and executes inside an isolated temporary directory with a timeout and network disabled.
   - Never invents numbers — every statistic, $p$-value, table, and figure comes directly from executed Python code.

4. **Write & Publish Canonical Manuscripts** (`agents/writing_publication.py`) — *“From data to publication.”*
   - Writes complete, canonically structured academic papers (`Title`, `Structured Abstract`, `Keywords & Highlights`, `1. Introduction`, `2. Literature Review`, `3. Materials & Methods`, `4. Results`, `5. Discussion`, `6. Limitations`, `7. Conclusion`, `CRediT Authorship`, `AI Disclosure`, and `Verified Reference List`) in **APA, IEEE, Harvard, MLA, or Vancouver** style.
   - Recommends suitable peer-reviewed journals via live **OpenAlex Sources & Crossref Journals** data with scope-fit reasoning, Open Access / DOAJ status, and links to author guidelines (never invents Impact Factors or acceptance rates).
   - Simulates double-anonymous peer review (**2 Reviewers + Associate Editor Decision**) and drafts a **Point-by-Point Response-to-Reviewers Letter**, **Submission Cover Letter**, and **Predatory-Journal Warning Checklist**.
   - Exports complete papers to **DOCX, PDF, and Markdown**.

5. **Get Mentored by the AI PhD Supervisor** (`agents/supervisor.py`) — *“Your always-available research mentor.”*
   - Builds a personalized 5-pillar research roadmap (**Plan → Guide → Feedback → Improve → Succeed**) with an interactive milestone progress bar.
   - Challenges assumptions in **Friendly Critical Review Mode**.
   - Generates a **12-Slide Presentation Outline**, **4-Quadrant Academic Poster Layout**, and **10 Mock Viva / Oral Defense Questions with Model Answers**.
   - Teaches academic ethics and provides a **24/7 Conversational Mentor Chat** with session memory.

6. **Create Impact** (`graph/workflow.py`) — *“Because better research creates a better world.”*
   - Orchestrates all 5 agents via a **LangGraph `StateGraph`** with **Human-Approval Checkpoints** between every stage.
   - Produces a **Plain-Language Summary**, **LinkedIn / Social Announcement Posts**, **Policy / Industry Brief**, **Press-Release Summary**, **Visibility Plan (ORCID, Google Scholar, Preprints, Conferences)**, and **Citation Tracking Checklist**.

---

## 2. Local Setup & Installation

```bash
# 1. Navigate to the project folder
cd research_ai

# 2. Create and activate a virtual environment (.venv)
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install all pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Configure your .env file
cp .env.example .env
# Edit .env to add your preferred API key (or paste keys directly in the app sidebar)

# 5. Launch the Streamlit application
streamlit run app.py
```

---

## 3. How to Get API Keys (Free & Paid Options)

ResearchAI supports **4 provider modes** out of the box. You can enter keys in the **Sidebar**, `.streamlit/secrets.toml`, or `.env`.

1. **Google Gemini (Free Tier Available)**
   - Visit [Google AI Studio (aistudio.google.com/app/apikey)](https://aistudio.google.com/app/apikey).
   - Sign in with your Google account and click **Create API Key**.
   - Models: `gemini-2.0-flash` (fast, generous free tier), `gemini-1.5-pro` (deep reasoning & long context).

2. **Groq (Fast Free Tier Available)**
   - Visit [Groq Console (console.groq.com/keys)](https://console.groq.com/keys).
   - Create a free account and generate an API key.
   - Models: `llama-3.3-70b-versatile` (high quality), `llama-3.1-8b-instant` (ultra-fast).

3. **OpenAI (Paid Credits)**
   - Visit [OpenAI Platform (platform.openai.com/api-keys)](https://platform.openai.com/api-keys).
   - Models: `gpt-4o-mini` (cost-effective), `gpt-4o`, `gpt-4.1`.

4. **Custom API (OpenAI-Compatible — Free Cloud & Local Models)**
   - **OpenRouter (Free Models)**: Get a key at [openrouter.ai/keys](https://openrouter.ai/keys), select the **OpenRouter** preset in the sidebar (`https://openrouter.ai/api/v1`), and use any `:free` model such as `meta-llama/llama-3.3-70b-instruct:free`.
   - **Ollama (100% Free & Local)**: Install [Ollama](https://ollama.com), run `ollama serve` and `ollama pull llama3.2`. Select the **Ollama** preset (`http://localhost:11434/v1`) — **no API key needed!**
   - **LM Studio (100% Free & Local)**: Start the Local Server on port `1234` (`http://localhost:1234/v1`).
   - **DeepSeek / Together AI / Mistral / xAI**: Select the corresponding preset in the sidebar and paste your API key.

---

## 4. Which Provider Should I Pick? (Comparison Table)

| Provider | Cost / Free Tier | Speed | Research & Writing Quality | Best Use Case in ResearchAI |
| :--- | :--- | :--- | :--- | :--- |
| **Groq (`llama-3.3-70b-versatile`)** | **Free tier** + low-cost pay-as-you-go | ⚡ Ultra-Fast (300+ tok/s) | ⭐⭐⭐⭐ (Excellent) | Rapid Discovery scans, PRISMA screening, and interactive Supervisor chat |
| **Google Gemini (`gemini-2.0-flash` / `1.5-pro`)** | **Free tier** in Google AI Studio | 🚀 Very Fast | ⭐⭐⭐⭐⭐ (Top-tier long context) | Full systematic literature reviews, multi-PDF synthesis & full manuscript writing |
| **OpenAI (`gpt-4o-mini` / `gpt-4o` / `gpt-4.1`)** | Paid API credits | 🚀 Fast | ⭐⭐⭐⭐⭐ (Gold standard) | Canonical journal manuscripts, statistical code generation & peer-review simulation |
| **Custom: OpenRouter (`*:free`)** | **Free** (`:free` models) | 🔄 Moderate | ⭐⭐⭐⭐ | Exploring diverse open-weight models (`Llama-3.3`, `Qwen`, `Mistral`) with one key |
| **Custom: Ollama / LM Studio (Local)** | **100% Free & Offline** | 💻 Depends on GPU/CPU | ⭐⭐⭐ to ⭐⭐⭐⭐ | Sensitive/unpublished lab data where 100% local privacy is required |

---

## 5. Step-by-Step Usage of Each Agent (with Example Inputs)

### Step 1 — AI Research Discovery Agent
- **Example Input:**
  - *Field:* `Agricultural & Environmental AI`
  - *Keywords:* `AI-based drought prediction for wheat using multi-spectral remote sensing`
  - *Career Stage:* `PhD Student / Candidate`
  - *Resources:* `University GPU workstation + public Sentinel-2 & agronomic trial datasets`
- **What Happens:** Queries arXiv, OpenAlex, Semantic Scholar, Crossref, and PubMed; renders interactive charts of publications per year, top authors, and top venues; identifies 4+ evidence-backed research gaps; and proposes **5 novel PhD research topics**.

### Step 2 — AI Literature Review Agent
- **Example Input:**
  - *Query:* `wheat drought stress prediction machine learning remote sensing`
  - *Inclusion Criteria:* `wheat, drought, machine learning, remote sensing, yield, empirical`
  - *Exclusion Criteria:* `corrigendum, erratum, book review`
  - *Optional:* Upload 1–5 PDF articles from your computer.
- **What Happens:** Deduplicates by DOI and normalized title, displays a **PRISMA Funnel Chart**, logs every inclusion/exclusion decision, extracts an **Evidence Comparison Matrix**, updates the **Verified Reference Library**, and writes a full systematic literature review.

### Step 3 — AI Research Experiment Agent
- **Example Input:**
  - Upload your `.csv` or `.xlsx` dataset (or select **Load Labeled Synthetic Demo Dataset** to test with `sample_data/wheat_drought_trial_demo.csv`).
- **What Happens:**
  1. Generates an **Experimental Design Protocol** ($H_0/H_1$, variables, controls, sample-size reasoning, validity threats).
  2. Writes a reproducible Python analysis script and **displays it in the code editor for your review**.
  3. Validates the script with an AST security scanner and waits for you to click **▶️ Run Python Analysis in Restricted Sandbox**.
  4. Displays exact `stdout`, generated CSV tables, publication PNG figures, and a draft Methods & Results section grounded 100% in the executed numbers.

### Step 4 — AI Academic Writing & Publication Agent
- **Example Input:**
  - *Title:* `Explainable Multi-Modal Machine Learning for Early Drought Stress and Grain Yield Prediction in Wheat (Triticum aestivum)`
  - *Citation Style:* `APA` (or `IEEE`, `Harvard`, `MLA`, `Vancouver`)
  - *Target Journal:* `Computers and Electronics in Agriculture`
- **What Happens:** Writes a complete canonical manuscript citing **only** the Verified Reference Library and reporting **only** the executed statistics from Agent 3. Also recommends journals via OpenAlex/Crossref, provides a predatory-journal checklist, simulates a 2-reviewer + editor peer review, and drafts a response-to-reviewers letter and cover letter.

### Step 5 — AI PhD Supervisor Agent
- **Example Input:**
  - *Topic:* `AI-based drought prediction in wheat`
  - *Horizon:* `3-Year PhD Journey (36 Months)`
- **What Happens:** Builds a 5-pillar roadmap (`Plan → Guide → Feedback → Improve → Succeed`), runs **Friendly Critical Review Mode** to challenge your assumptions, generates a **12-Slide Presentation Outline + Poster Layout + 10 Mock Viva Q&As**, and offers a **24/7 Mentor Chat**.

### Step 6 — Full LangGraph Workflow & Create Impact
- Use the **Checkpoint Controller** to step through Agents 1 → 6 with human approval at each stage, or click **Generate Step 6 Impact & Outreach Package** to create your Lay Summary, LinkedIn Post, Policy Brief, Press Release, ORCID/Google Scholar Visibility Plan, and Citation Tracking Checklist.

---

## 6. Data Format Guide & Safe Code Runner Architecture (Experiment Agent)

### Preparing Your CSV / Excel File
- **Row 1** should contain clean column headers (e.g., `variety`, `drought_treatment`, `soil_moisture_pct`, `ndvi_score`, `grain_yield_t_ha`).
- Include at least **1 categorical/grouping column** (e.g., treatment group, cohort, species) and **2+ numeric measurement columns** for automatic ANOVA/Kruskal-Wallis hypothesis testing, correlation heatmaps, OLS regression, and Random Forest cross-validation.

### How the Restricted Subprocess Code Runner Works (`utils/code_runner.py`)
1. **Pre-Execution AST Validation**: Parses the Python code into an Abstract Syntax Tree (`ast.parse`) and blocks any import outside the scientific allow-list (`pandas`, `numpy`, `scipy`, `statsmodels`, `sklearn`, `matplotlib`, `seaborn`, `plotly`, `Bio`). Dangerous modules (`os`, `sys`, `subprocess`, `socket`, `requests`, `shutil`, `pickle`) and builtins (`eval`, `exec`, `__import__`) are rejected before execution.
2. **Human-in-the-Loop Preview**: The full Python script is shown in an editable code box. Nothing runs until you explicitly click **Run**.
3. **Ephemeral Temp Sandbox**: Runs in a fresh `tempfile.TemporaryDirectory()` where network sockets (`socket.socket`) are disabled and `open()` is restricted to the temporary folder.
4. **Timeout Protection**: Enforces a hard 25-second subprocess timeout.

---

## 7. Deploying to GitHub & Streamlit Community Cloud

1. **Push to GitHub**:
   - Ensure `.gitignore` is present so `.venv`, `.env`, and `.streamlit/secrets.toml` are **never** committed.
   - Push the repository to your GitHub account.
2. **Deploy on [share.streamlit.io](https://share.streamlit.io)**:
   - Click **New app** in Streamlit Community Cloud.
   - Select your GitHub repository, branch (`main`), and main file path (`research_ai/app.py` or `app.py`).
   - Open **Advanced Settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example` with your API keys:
     ```toml
     OPENAI_API_KEY = "sk-..."
     GOOGLE_API_KEY = "AIza..."
     GROQ_API_KEY = "gsk_..."
     ```
   - Click **Deploy!**

---

## 8. Troubleshooting & Known Limitations

- **Free-Tier Rate Limits (HTTP 429):** Free tiers on Groq, Gemini, and OpenRouter have requests-per-minute limits. Wait 30–60 seconds, reduce "Papers per Database" to 3–4, or switch to another provider in the sidebar.
- **Local Ollama / LM Studio on Cloud:** `http://localhost:11434/v1` works when running Streamlit on your own computer. When deployed on Streamlit Community Cloud, use a cloud provider (OpenAI, Gemini, Groq, OpenRouter, Together, DeepSeek).
- **Always Verify Citations & Outputs:** While ResearchAI restricts references to real API records, metadata in public registries can occasionally have missing fields or preprint versions. Always click the DOI link and read the primary paper before submitting a manuscript.

---

## 9. Responsible AI Use in Academic Research

1. **Follow Journal & University Policies:** Major publishers (Nature, Science, IEEE, Elsevier, Springer, Wiley, PLOS) permit AI tools for literature organization, code assistance, and language polishing **provided their use is transparently disclosed**.
2. **Mandatory AI Disclosure:** Include the AI-Use Disclosure generated in Section 8 of the Writing Agent in your manuscript's Declarations section.
3. **AI Cannot Be an Author:** Under ICMJE and COPE guidelines, Large Language Models cannot take legal or ethical accountability and **must never be listed as co-authors**.
4. **Human Researcher Accountability:** You (the researcher) and your human academic supervisor are 100% accountable for every claim, citation, statistical test, ethical approval (IRB/IACUC), and conclusion in your work.

---
*“Higher Research, Brighter Tomorrow.” — ResearchAI*
