from ai_assistant_chain import search_corpus
from memory_store import search_memory, write_memory
from mcp_clients import run_mcp_tool


def _normalize_sources(enabled_sources):
    """
    Normalize enabled sources to lowercase values.

    Args:
        enabled_sources (list): Source names selected in UI.

    Returns:
        list: Normalized source names.
    """
    if not enabled_sources:
        return ["notion", "jira", "github"]
    return [source.lower() for source in enabled_sources]


def get_agent_tools(enabled_sources, session_id="default", memory_enabled=False, memory_top_k=3):
    """
    Build the list of agent tools for the current query.

    Args:
        enabled_sources (list): Source names selected in UI.

    Args:
        session_id (str, optional): Active user session id.
        memory_enabled (bool, optional): Enable memory tools for this session.
        memory_top_k (int, optional): Memory retrieval top-k.

    Returns:
        list: Tool descriptors with callable handlers.
    """
    selected_sources = _normalize_sources(enabled_sources)
    tools = []

    def _search_corpus_tool(payload):
        query = payload.get("query", "")
        source_list = payload.get("sources", selected_sources)
        top_k = int(payload.get("top_k", 4))
        return search_corpus(query=query, source_list=source_list, top_k=top_k)

    tools.append({
        "name": "search_corpus",
        "description": "Searches local FAISS corpus chunks by source filter.",
        "read_only": True,
        "run": _search_corpus_tool
    })

    if memory_enabled:
        tools.append({
            "name": "search_memory",
            "description": "Searches session memory records with TTL and dedupe.",
            "read_only": True,
            "run": lambda payload: search_memory(
                query=payload.get("query", ""),
                session_id=payload.get("session_id", session_id),
                top_k=int(payload.get("top_k", memory_top_k))
            )
        })

        tools.append({
            "name": "write_memory",
            "description": "Writes sanitized session memory records (post-answer only).",
            "read_only": False,
            "run": lambda payload: write_memory(
                memory_text=payload.get("memory_text", ""),
                session_id=payload.get("session_id", session_id),
                tags=payload.get("tags", []),
                ttl_days=int(payload.get("ttl_days", 30))
            )
        })

    if "github" in selected_sources:
        tools.append({
            "name": "mcp_github_read",
            "description": "Reads GitHub issues, pull requests, and recent commits.",
            "read_only": True,
            "run": lambda payload: run_mcp_tool("github", "read_release_state", payload)
        })

    if "jira" in selected_sources:
        tools.append({
            "name": "mcp_jira_read",
            "description": "Reads Jira issues and status for release readiness.",
            "read_only": True,
            "run": lambda payload: run_mcp_tool("jira", "read_release_state", payload)
        })

    if "notion" in selected_sources:
        tools.append({
            "name": "mcp_notion_read",
            "description": "Reads Notion pages relevant to release plans and docs.",
            "read_only": True,
            "run": lambda payload: run_mcp_tool("notion", "read_release_state", payload)
        })

    return tools
