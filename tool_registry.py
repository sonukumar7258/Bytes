from ai_assistant_chain import search_corpus
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


def get_agent_tools(enabled_sources):
    """
    Build the list of agent tools for the current query.

    Args:
        enabled_sources (list): Source names selected in UI.

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
