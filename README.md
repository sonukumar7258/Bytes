## Demo Video




https://github.com/ByteCorp-Tech/2024-AI-Challenge-Miraculum-generationis/assets/28065322/edb7520a-a535-4a58-a09c-6fc063eb3ce3





# Bytes

Bytes is a chat assistant designed to streamline access to various data sources within your company, including Jira, Github, Notion, and your company's website. Users can also upload their own files and leverage the capabilities of the Assistant for them. It is built using Python, with Solara for UI and Langchain for Retrieval Augmented Generation.

## Features
- **Multi-source Data Retrieval**: Access data from Jira, Github, Notion, and your company's website seamlessly within the chat interface. Users can enable/disable specific sources as per their preference.
- **Natural Language Understanding**: Utilizes advanced NLP techniques to understand user queries and provide relevant responses.
- **Intuitive UI**: Built on Solara for a user-friendly interface, making interaction with the assistant smooth and intuitive.
- **Augmented Generation**: Powered by Langchain, the assistant not only retrieves data but also generates augmented responses, enhancing user experience.
- **Multiple Open Source Models**: Supports multiple open source models from the Llama Family by Meta, Qwen (`qwen/qwen3-32b`), and Kimi (`moonshotai/kimi-k2-instruct`) via Groq, allowing users to switch between LLMs based on their requirements.
- **Source Information URLs**: Provides URLs for the source information the assistant uses for its answer to promote transparency and traceability.
- **Agentic Mode (RAG + MCP-style tools)**: Adds an orchestrated agent loop that can route between local corpus retrieval and live read-only tools for GitHub, Jira, and Notion.
- **Release Readiness Brief**: Specialized agent workflow to summarize release status, blockers, risks, and next actions with evidence links.
- **Tool Timeline Transparency**: UI panel shows planning/tool/synthesis steps with status and duration for interview walkthroughs.


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
This document outlines the methodology and approach of Bytes, which leverages Large Language Models (LLMs) to process and respond to various data sources like Jira, GitHub, Notion, and general websites. The system is designed to intelligently manage and utilize data across these platforms, enhancing retrieval capabilities and embedding management.

## Components
Bytes is composed of several key components that interact to process data, generate embeddings, and facilitate intelligent query handling:

### Data Processing
- **Corpus Loading**: JSON files representing different data sources (e.g., Notion, Jira, GitHub) are loaded into the system.
- **Data Cleaning**: Data from these sources is cleaned and structured. Specific keys are removed from Notion data to refine the content.
- **Data Flattening and Parsing**: Complex data structures from Jira and GitHub are flattened, and website data is chunked into manageable pieces for further processing.

### Embedding Management
- **Embedding Generation**: Utilizes `OpenAIEmbeddings` to transform processed text data into dense vector representations.
- **FAISS Indexing**: Embeddings are stored in FAISS indices, a highly efficient similarity search library, which allows for quick retrieval of related documents based on vector similarity.

### Query Processing
- **Chunking**: Large texts are split into smaller chunks to manage the load on the LLM and improve the response accuracy.
- **Embedding Retrieval**: For a given query, the system retrieves the most relevant embeddings from the FAISS store.
  
### Integration with LangChain
- **Chains and Prompts**: The system uses LangChain to create sophisticated chains of operations, such as data retrieval chains and document processing chains.
- **LLM Integration**: The assistant integrates with various LLMs like Google Generative AI and open source LLMs from Llama family, Qwen, and Kimi for processing and generating responses based on the context provided by the embeddings.

### Response Generation
- **Dynamic LLM Switching**: Depending on user preferences, the assistant can switch between different LLM configurations to generate the most accurate and contextually appropriate responses.
- **PDF Processing**: Capable of extracting text from PDF files, processing the content, and loading it into the vector store for query handling. 

## Operational Flow
1. **Initialization**: Load environment variables and initialize embeddings.
2. **Data Loading**: Load and process JSON corpora from multiple sources.
3. **Embedding Storage**: Generate and store embeddings in FAISS.
4. **Query Handling**: On receiving a query, chunk the necessary texts, retrieve relevant embeddings, and generate a response using the selected LLM.
5. **Output**: The system formats and delivers the response, handling any follow-up queries by referencing the stored context and embeddings.

## Advanced Features
- **Parallel Processing**: The system is designed to handle multiple tasks in parallel, significantly speeding up data processing and response generation.
- **Configurable Runtime**: Depending on the operational needs, different components of the AI assistant can be configured dynamically, allowing flexible adaptations to various types of queries and data sources.

This methodology ensures a robust, scalable, and efficient AI assistant capable of handling complex queries across multiple domains, utilizing advanced AI and machine learning techniques to enhance productivity and decision-making processes.
