# Bytes

Bytes is a hybrid AI assistant designed to streamline access to company knowledge across Jira, GitHub, Notion, and internal website content. Users can also upload PDFs for ad-hoc Q&A. It is built in Python with Solara UI, LangChain retrieval, and an agentic runtime that can orchestrate runtime tool calls.

## Features
- **Multi-source Data Retrieval**: Access data from Jira, Github, Notion, and your company's website seamlessly within the chat interface. Users can enable/disable specific sources as per their preference.
- **Natural Language Understanding**: Utilizes advanced NLP techniques to understand user queries and provide relevant responses.
- **Intuitive UI**: Built on Solara for a user-friendly interface, making interaction with the assistant smooth and intuitive.
- **Hybrid Retrieval + Agentic Orchestration**: Combines FAISS-based RAG with agentic tool orchestration for live, freshness-sensitive queries.
- **Multiple Open Source Models**: Supports multiple open source models from the Llama Family by Meta, Qwen (`qwen/qwen3-32b`), and Kimi (`moonshotai/kimi-k2-instruct`) via Groq, allowing users to switch between LLMs based on their requirements.
- **Source Information URLs**: Provides URLs for the source information the assistant uses for its answer to promote transparency and traceability.
- **Agentic Mode (RAG + MCP-style tools)**: Adds an orchestrated agent loop that can route between local corpus retrieval and live read-only tools for GitHub, Jira, and Notion.
- **Release Readiness Brief**: Specialized agent workflow to summarize release status, blockers, risks, and next actions with evidence links.
- **Tool Timeline Transparency**: UI panel shows planning/tool/synthesis steps with status and duration for interview walkthroughs.

## Why RAG-only is not enough here
- **RAG is snapshot-based**: A local vector store is excellent for stable documentation, but it can lag behind current Jira/GitHub/Notion state unless constantly re-indexed.
- **Operational questions are cross-system**: Queries like release readiness require combining multiple systems (tickets, PRs, commits, docs), not just nearest text chunks.
- **Freshness matters**: Questions containing "latest", "recent", or "today" are better answered via runtime reads from live sources.
- **Traceability matters**: Agentic mode provides tool timeline, confidence, and citation output so you can explain how the answer was produced.
- **Best of both worlds**: RAG remains the fast, grounded path for static/internal knowledge; agentic mode is used when orchestration and live evidence are required.


**Installation and Execution Guide**
=====================================

**Step 1: Create Environment Variables**
------------------------------------

After cloning the repository, create a `.env` file in the project directory with the following environment variables:

* `JIRA_API_TOKEN`
* `OPENAI_API_KEY`
* `JIRA_USERNAME`
* `JIRA_INSTANCE_URL`
* `GOOGLE_API_KEY`
* `GITHUB_ACCESS_TOKEN`
* `NOTION_API_KEY`
* `GROQ_API_KEY`

**Obtaining Environment Variables**

* `JIRA_API_TOKEN` and `JIRA_USERNAME` can be obtained from any Jira company account with read access.
* `OPENAI_API_KEY` can be obtained from the OpenAI API website.
* `GITHUB_ACCESS_TOKEN` can be obtained from any GitHub account.
* `NOTION_API_KEY` can be obtained by creating an integration in Notion as an admin and using the secret key. Integrate the integration in all Notion pages you want to retrieve data from.
* `GROQ_API_KEY` can be obtained from Groq Cloud (for LLaMA, Qwen, and Kimi models).

**Step 2: Create Conda Environment**
------------------------------------

Create a Conda environment using the `environment.yml` file in the project directory:

```
conda env create -f environment.yml
conda activate <env_name>
```

**Step 3: Run Data Extraction Scripts**
--------------------------------------

Run the following scripts to extract data from various sources:

```
python data_extraction/extract_corpus_notion.py
python data_extraction/extract_corpus_github.py
python data_extraction/extract_corpus_jira.py
```

**Step 4: Create and Store Vector Stores and Embeddings**
---------------------------------------------------

Run the following script to create and store vector stores and embeddings for all data sources locally:

```
python save_vector_store.py
```

**Step 5: Run the Assistant**
-------------------------

Run the assistant using Solara:

```
solara run ai_assistant.py
```

That's it! You should now have the chatbot up and running.

**Step 6: Run Agentic Evaluation Pack (Optional)**
--------------------------------------

Run deterministic interview scenarios and export a markdown demo report:

```
python evaluate_agentic.py --model gemini --sources notion jira github --output agentic_demo_report.md
```

This report tracks tool-choice correctness, citation presence, freshness behavior, and hallucination heuristics.


# Methodology and approach

## Overview
Bytes now follows a **hybrid architecture**:
- **RAG mode** for high-precision answers from local indexed knowledge.
- **Agentic mode** for runtime tool orchestration across Jira, GitHub, and Notion with citation transparency.

This makes the assistant suitable for both:
- stable documentation Q&A (RAG),
- freshness-sensitive operational queries like release readiness (Agentic).

## Components
Bytes is composed of modular components that support both retrieval and agent orchestration.

### Data Processing
- **Corpus Extraction Scripts**: Source-specific scripts fetch data from Notion, GitHub, and Jira and store it as JSON.
- **Flattening and Normalization**: Helper utilities convert nested API payloads into searchable textual chunks with metadata.
- **Chunking**: Text is chunked before embedding to improve retrieval quality.

### Embedding Management
- **Embedding Generation**: `OpenAIEmbeddings` transforms chunks into vectors.
- **FAISS Vector Store**: Embeddings are stored locally and loaded at runtime for retrieval.
- **Source Filtering**: Documents retain source metadata (`notion`, `jira`, `github`, `website`) used for filter-aware retrieval.

### RAG Runtime
- **Retrieval Chain**: LangChain retrieval chain fetches relevant chunks from FAISS.
- **Strict Context Prompting**: Primary prompt enforces context-only answers and rejects unsupported responses.
- **Fallback Context Synthesis**: If strict mode returns "no context" despite retrieved chunks, a fallback context-grounded answer pass is used.
- **URL Extraction and Ranking**: Related links are deduplicated and ranked before rendering.

### Agentic Runtime
- **Intent Classification**: Query is classified into categories such as knowledge lookup, freshness lookup, release readiness, or write-like intent.
- **Strategy Selection**: Runtime chooses `corpus_first`, `live_first`, `hybrid`, or `reject` (for write requests in read-only mode).
- **Tool Execution Layer**:
  - `search_corpus` for local retrieval,
  - `mcp_jira_read`, `mcp_github_read`, `mcp_notion_read` for live reads.
- **MCP-style Client Wrapper**: External tool calls include allowlisting, retries, and timeout control.
- **Synthesis and Citations**: Final answer is generated from collected evidence with links, confidence, and an optional timeline of steps.
- **Guardrails**: Write-like actions are blocked by policy and converted into safe read-only responses.

### UI and Interaction
- **Mode Switching**: User can switch between `rag` and `agentic` modes from the Solara sidebar.
- **Model Switching**: Runtime model switching supports Gemini, Llama family, Qwen, and Kimi options.
- **Tool Timeline View**: Agentic mode can show planning/tool/synthesis steps with durations.
- **PDF Path**: Uploaded PDF creates an in-memory/local vector retrieval chain for ad-hoc Q&A.

## Operational Flow
1. **Startup**
- Load environment variables and initialize embeddings + default LLM.
- Load FAISS vector store and prepare retrieval chain.

2. **User Query**
- User selects model, data sources, and assistant mode (RAG or Agentic).

3. **Execution Path**
- **RAG mode**: retrieve -> answer -> fallback (if needed) -> link ranking.
- **Agentic mode**: classify -> decide strategy -> call tools -> synthesize -> cite -> timeline.

4. **Response Rendering**
- UI displays answer text, related links, optional timeline, and confidence metadata (agentic path).

## Evaluation and Interview Readiness
- **Deterministic Scenario Pack**: `evaluate_agentic.py` runs structured prompts that validate routing/tool behavior.
- **Scoring Dimensions**:
  - tool-choice correctness,
  - citation presence,
  - freshness correctness,
  - hallucination heuristics.
- **Report Export**: Generates a markdown report for interview demonstration.

This methodology enables Bytes to operate as a practical production-style assistant: reliable for static knowledge retrieval while also capable of runtime, multi-tool reasoning for dynamic business questions.
