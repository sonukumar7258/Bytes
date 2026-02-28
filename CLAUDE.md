# CLAUDE.md — Bytes AI Assistant

## Project Purpose

**Bytes** is a multi-source RAG (Retrieval-Augmented Generation) chat assistant built for the 2024 ByteCorp AI Challenge. It allows users to query company knowledge from Jira, GitHub, Notion, and internal websites using natural language. Users can also upload PDFs for ad-hoc queries.

**Start the app:**
```bash
solara run ai_assistant.py
```

---

## Architecture

### RAG Pipeline
```
Data Sources (Jira, GitHub, Notion, Website)
    ↓
Data Extraction (data_extraction/extract_corpus_*.py)
    ↓
Text Flattening & Chunking (utilities/ + save_vector_store.py)
    ↓
OpenAI Embeddings → FAISS Vector Store (embeddings/all/)
    ↓
LangChain Retrieval Chain
    ↓
LLM Response (Gemini / LLaMA / Mixtral / Gemma)
    ↓
Solara Web UI (ai_assistant.py)
```

### Key Architectural Decisions

- **Pre-built vector store**: FAISS index is built once via `save_vector_store.py` and loaded at runtime. It is never rebuilt during a session.
- **Global state for LLM and chain**: `ai_assistant_chain.py` holds the active LLM and retrieval chain as module-level globals, mutated by `switch_llm()` and `load_vector_store()`.
- **Source metadata filtering**: Documents are tagged with `source:<name>` metadata. FAISS `search_kwargs` filter documents by selected sources at query time.
- **Multi-LLM runtime switching**: Users switch between Gemini (default), LLaMA 3 8B/70B, Mixtral 8x7B, and Gemma 7B without restarting.
- **Threading**: Queries run in a background thread (`threading.Thread`) to keep the Solara UI responsive.
- **Strict prompt discipline**: The system prompt instructs the LLM to answer only from retrieved context. If no context is found, it replies with a canned "no context" message to prevent hallucination.
- **Source transparency**: URLs are extracted from retrieved document metadata and surfaced to the user after each response.
- **Chunk size**: 1500 characters with 300-character overlap (configured in `save_vector_store.py`).

---

## Key Files

| File | Responsibility |
|------|----------------|
| `ai_assistant.py` | Solara UI — reactive state, sidebar controls, chat interface, event handlers |
| `ai_assistant_chain.py` | LangChain RAG chain, LLM switching, PDF loading, URL extraction from context |
| `save_vector_store.py` | One-time setup: loads corpora, chunks text, creates embeddings, saves FAISS index |
| `data_extraction/extract_corpus_notion.py` | Fetches all Notion pages recursively via Notion API |
| `data_extraction/extract_corpus_github.py` | Fetches repos, branches, commits, and issues via GitHub API |
| `data_extraction/extract_corpus_jira.py` | Fetches projects, issues, and comments via Jira API |
| `utilities/notion_helper_functions.py` | Parses Notion block dicts into flat text with URL metadata |
| `utilities/github_helper_functions.py` | Flattens GitHub issue/commit data into text chunks |
| `utilities/jira_helper_functions.py` | Flattens Jira corpus into text chunks with metadata |
| `utilities/website_helper_functions.py` | Chunks website text and reconstructs source URLs |

**Generated artifacts (gitignored — do not commit):**
- `corpus/*.json` — raw extracted data
- `embeddings/all/` — FAISS vector index
- `context.txt` — last query's retrieved context (useful for debugging)
- `website_data/` — scraped website content

---

## Coding Conventions

- **Language**: Python 3.11
- **Style**: snake_case for all functions and variables
- **Type hints**: Not used in this codebase — do not add them
- **Docstrings**: All public functions have docstrings with `Args:` and `Returns:` sections — maintain this pattern
- **Global mutations**: Always declare `global <var>` before mutating module-level state
- **Imports order**: stdlib → `dotenv` → `langchain` → source-specific API clients
- **Relative imports**: `from utilities.<source>_helper_functions import <function>`
- **File I/O**: Always use `with open(...)` context managers
- **Environment variables**: Load via `python-dotenv`; never hardcode credentials

---

## Common Tasks

### Debugging

1. **Blank or wrong answers** — Inspect `context.txt` after a query to see what was retrieved. If it's empty, the vector store may not contain relevant content or source filters are too restrictive.
2. **FAISS load error** — `embeddings/all/` doesn't exist. Run `python save_vector_store.py` to rebuild.
3. **API key errors** — Verify `.env` contains all required keys (see below).
4. **Solara UI state issues** — Check reactive variable declarations in `ai_assistant.py`. Ensure `global` is declared in any function that mutates chain-level state.
5. **API rate limits** — Each extraction script in `data_extraction/` can be run independently and re-run without affecting others.

### Adding a New Data Source

1. Create `data_extraction/extract_corpus_<source>.py` — follow the existing extraction scripts as a template
2. Create `utilities/<source>_helper_functions.py` — implement `flatten_corpus(corpus)` returning a list of text strings with source URL metadata
3. In `save_vector_store.py`: load the new corpus JSON, flatten it, and include it in the documents list before embedding
4. Tag documents with `metadata={"source": "<source>", "url": "..."}` — the source tag is used for filtering
5. In `ai_assistant.py`: add a checkbox toggle and wire it to `on_value_change_tools()`
6. In `ai_assistant_chain.py` → `load_vector_store()`: add the new source to the filter logic

### Adding or Switching an LLM

1. In `ai_assistant_chain.py` → `switch_llm()`: add a new `elif name == "<model>"` branch instantiating the LLM
2. In `ai_assistant.py`: add the model name string to the model selector dropdown
3. Add the provider's API key to `.env` and load it with `os.getenv()`

### Rebuilding the Vector Store

```bash
python data_extraction/extract_corpus_notion.py
python data_extraction/extract_corpus_github.py
python data_extraction/extract_corpus_jira.py
python save_vector_store.py
```

Then restart `solara run ai_assistant.py`.

### Full Local Setup

```bash
conda env create -f environment.yml
conda activate <env-name>
# Create .env and fill in all keys (see below)
python data_extraction/extract_corpus_notion.py
python data_extraction/extract_corpus_github.py
python data_extraction/extract_corpus_jira.py
python save_vector_store.py
solara run ai_assistant.py
```

---

## Required Environment Variables (`.env`)

```
OPENAI_API_KEY        # Used for embeddings (required always)
GOOGLE_API_KEY        # Gemini models
GROQ_API_KEY          # LLaMA 3, Mixtral, Gemma via Groq
NOTION_API_KEY        # Notion data extraction
GITHUB_ACCESS_TOKEN   # GitHub data extraction
JIRA_API_TOKEN        # Jira data extraction
JIRA_USERNAME         # Jira data extraction
JIRA_INSTANCE_URL     # Jira data extraction (e.g. https://yourorg.atlassian.net)
```
