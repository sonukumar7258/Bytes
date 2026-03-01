from dotenv import load_dotenv
from datetime import datetime, timezone
import time

from ai_assistant_chain import get_active_llm, switch_llm
from guardrails import (
    check_input_guardrail,
    check_output_citation_gate,
    extract_memory_candidate,
    sanitize_memory_text
)
from tool_registry import get_agent_tools

load_dotenv()

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None


WRITE_TERMS = [
    "create",
    "update",
    "delete",
    "close",
    "reopen",
    "assign",
    "comment on",
    "post to",
    "edit"
]

FRESHNESS_TERMS = [
    "latest",
    "today",
    "recent",
    "right now",
    "current",
    "newest",
    "open now"
]

RELEASE_TERMS = [
    "release",
    "readiness",
    "go live",
    "ship",
    "cutover",
    "blocker",
    "launch"
]


def _utc_now_iso():
    """
    Return current UTC timestamp in ISO format.

    Returns:
        str: Current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _ensure_model(model_name):
    """
    Ensure the requested model is active before answering.

    Args:
        model_name (str): Selected UI model identifier.
    """
    if model_name == "gemini":
        switch_llm("gemini", "gemini-pro")
    elif model_name == "qwen/qwen3-32b":
        switch_llm("llama", "qwen/qwen3-32b")
    elif model_name == "llama3-8b-8192":
        switch_llm("llama", "llama-3.1-8b-instant")
    elif model_name == "llama3-70b-8192":
        switch_llm("llama", "llama-3.3-70b-versatile")
    elif model_name == "moonshotai/kimi-k2-instruct":
        switch_llm("llama", "moonshotai/kimi-k2-instruct")
    elif model_name == "GPT5.2":
        switch_llm("gpt", "gpt-5.2")


def _append_timeline(state, step_type, tool, input_summary, result_summary, started_at, status="ok"):
    """
    Append one timeline event into state.

    Args:
        state (dict): Runtime state.
        step_type (str): Step category.
        tool (str): Tool or node name.
        input_summary (str): Step input summary.
        result_summary (str): Step output summary.
        started_at (float): Start timestamp.
        status (str, optional): Step status.
    """
    duration_ms = int((time.time() - started_at) * 1000)
    state["timeline"].append({
        "step_type": step_type,
        "tool": tool,
        "input_summary": input_summary,
        "result_summary": result_summary,
        "duration_ms": duration_ms,
        "status": status,
        "timestamp": _utc_now_iso()
    })


def _normalize_enabled_sources(enabled_sources):
    """
    Normalize enabled source names.

    Args:
        enabled_sources (list): Source names from UI.

    Returns:
        list: Lowercase source names.
    """
    if not enabled_sources:
        return ["notion", "jira", "github"]
    return [source.lower() for source in enabled_sources]


def _classify_intent_node(state):
    """
    Classify query intent and store it in state.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    query = (state.get("query") or "").lower()

    has_write = any(term in query for term in WRITE_TERMS)
    has_freshness = any(term in query for term in FRESHNESS_TERMS)
    has_release = any(term in query for term in RELEASE_TERMS)

    if has_write:
        intent = "write_request"
    elif has_release:
        intent = "release_readiness"
    elif has_freshness:
        intent = "freshness_lookup"
    else:
        intent = "knowledge_lookup"

    state["intent"] = intent
    _append_timeline(
        state,
        step_type="plan",
        tool="classify_intent",
        input_summary=f"query={state.get('query', '')[:90]}",
        result_summary=f"intent={intent}",
        started_at=started_at
    )
    return state


def _input_guardrail_node(state):
    """
    Evaluate user input for prompt-injection patterns.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    result = check_input_guardrail(state.get("query", ""))
    state["input_guardrail"] = result
    if result.get("triggered"):
        state["guardrail_flags"].append("input_guardrail")
    if result.get("severity") == "high":
        state["force_safe_response"] = True

    _append_timeline(
        state,
        step_type="guardrail",
        tool="input_guardrail",
        input_summary=f"query={state.get('query', '')[:90]}",
        result_summary=f"triggered={result.get('triggered')}, severity={result.get('severity')}",
        started_at=started_at,
        status="ok" if not result.get("triggered") else "warning"
    )
    return state


def _retrieve_memory_node(state):
    """
    Retrieve session memory before selecting strategy.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    if not state.get("memory_enabled"):
        _append_timeline(
            state,
            step_type="memory",
            tool="search_memory",
            input_summary="memory disabled",
            result_summary="memory retrieval skipped",
            started_at=started_at,
            status="skipped"
        )
        return state

    memory_result = _run_tool(
        state,
        "search_memory",
        {
            "query": state.get("query", ""),
            "session_id": state.get("session_id", "default"),
            "top_k": state.get("memory_top_k", 3)
        }
    )
    state["memory_snippets"] = memory_result.get("snippets", [])
    state["memory_items"] = memory_result.get("items", [])
    state["memory_hit"] = len(state["memory_items"]) > 0

    _append_timeline(
        state,
        step_type="memory",
        tool="retrieve_memory",
        input_summary=f"session_id={state.get('session_id', 'default')}",
        result_summary=f"memory_hits={len(state['memory_items'])}",
        started_at=started_at
    )
    return state


def _decide_strategy_node(state):
    """
    Decide execution strategy based on classified intent.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    intent = state.get("intent")

    if state.get("force_safe_response"):
        strategy = "reject_guardrail"
    elif intent == "write_request":
        strategy = "reject"
    elif intent == "freshness_lookup":
        strategy = "live_first"
    elif intent == "release_readiness":
        strategy = "hybrid"
    else:
        strategy = "corpus_first"

    state["strategy"] = strategy
    _append_timeline(
        state,
        step_type="plan",
        tool="decide_strategy",
        input_summary=f"intent={intent}",
        result_summary=f"strategy={strategy}",
        started_at=started_at
    )
    return state


def _tool_item_count(payload):
    """
    Return approximate item count for normalized tool payload.

    Args:
        payload (dict): Tool output payload.

    Returns:
        int: Number of returned records.
    """
    if not payload:
        return 0
    if "items" in payload and isinstance(payload.get("items"), list):
        return len(payload["items"])
    if "snippets" in payload and isinstance(payload.get("snippets"), list):
        return len(payload["snippets"])
    return 0


def _run_tool(state, tool_name, payload):
    """
    Execute one tool call with per-query circuit breaker and trace logging.

    Args:
        state (dict): Runtime state.
        tool_name (str): Registered tool name.
        payload (dict): Tool payload.

    Returns:
        dict: Tool result payload.
    """
    started_at = time.time()
    circuit_breaker = state["circuit_breaker"]
    if circuit_breaker.get(tool_name, 0) >= 1:
        _append_timeline(
            state,
            step_type="tool",
            tool=tool_name,
            input_summary="circuit breaker open",
            result_summary="skipped due to earlier failure",
            started_at=started_at,
            status="skipped"
        )
        return {}

    tool_def = state["tool_map"].get(tool_name)
    if not tool_def:
        _append_timeline(
            state,
            step_type="tool",
            tool=tool_name,
            input_summary=str(payload)[:100],
            result_summary="tool not enabled in current source selection",
            started_at=started_at,
            status="skipped"
        )
        return {}

    try:
        result = tool_def["run"](payload)
        state["tool_outputs"][tool_name] = result
        _append_timeline(
            state,
            step_type="tool",
            tool=tool_name,
            input_summary=str(payload)[:120],
            result_summary=f"success with {_tool_item_count(result)} records",
            started_at=started_at
        )
        return result
    except Exception as exc:
        circuit_breaker[tool_name] = circuit_breaker.get(tool_name, 0) + 1
        state["tool_errors"][tool_name] = str(exc)
        _append_timeline(
            state,
            step_type="tool",
            tool=tool_name,
            input_summary=str(payload)[:120],
            result_summary=f"failed: {exc}",
            started_at=started_at,
            status="error"
        )
        return {}


def _live_tool_names(state):
    """
    Return enabled live MCP tool names for the query.

    Args:
        state (dict): Runtime state.

    Returns:
        list: Live tool names.
    """
    ordered_live_tools = ["mcp_jira_read", "mcp_github_read", "mcp_notion_read"]
    return [name for name in ordered_live_tools if name in state["tool_map"]]


def _retrieve_or_tool_call_node(state):
    """
    Run retrieval and/or live tools following strategy and fallback rules.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    query = state.get("query", "")
    sources = state.get("enabled_sources", [])
    strategy = state.get("strategy", "hybrid")

    if strategy in ["reject", "reject_guardrail"]:
        _append_timeline(
            state,
            step_type="plan",
            tool="retrieve_or_tool_call",
            input_summary=f"strategy={strategy}",
            result_summary="execution stopped by safety policy",
            started_at=started_at,
            status="skipped"
        )
        return state

    live_tools = _live_tool_names(state)
    live_item_total = 0
    corpus_snippet_total = 0

    if strategy in ["corpus_first", "hybrid"]:
        corpus_result = _run_tool(
            state,
            "search_corpus",
            {"query": query, "sources": sources, "top_k": 4}
        )
        corpus_snippet_total = len(corpus_result.get("snippets", []))

    if strategy in ["live_first", "hybrid"]:
        for live_tool in live_tools:
            tool_result = _run_tool(
                state,
                live_tool,
                {"query": query, "item_limit": 6}
            )
            live_item_total += len(tool_result.get("items", []))

    if strategy == "live_first" and live_item_total == 0:
        corpus_result = _run_tool(
            state,
            "search_corpus",
            {"query": query, "sources": sources, "top_k": 4}
        )
        corpus_snippet_total = len(corpus_result.get("snippets", []))

    if strategy == "corpus_first" and corpus_snippet_total == 0:
        for live_tool in live_tools:
            tool_result = _run_tool(
                state,
                live_tool,
                {"query": query, "item_limit": 6}
            )
            live_item_total += len(tool_result.get("items", []))

    _append_timeline(
        state,
        step_type="plan",
        tool="retrieve_or_tool_call",
        input_summary=f"strategy={strategy}",
        result_summary=f"corpus_snippets={corpus_snippet_total}, live_items={live_item_total}",
        started_at=started_at
    )
    return state


def _collect_evidence(state):
    """
    Collect text evidence from tool outputs for synthesis.

    Args:
        state (dict): Runtime state.

    Returns:
        tuple: (corpus_snippets, live_items, memory_snippets)
    """
    corpus_result = state["tool_outputs"].get("search_corpus", {})
    corpus_snippets = corpus_result.get("snippets", [])

    live_items = []
    for live_tool in _live_tool_names(state):
        payload = state["tool_outputs"].get(live_tool, {})
        live_items.extend(payload.get("items", []))
    memory_snippets = state.get("memory_snippets", [])
    return corpus_snippets, live_items, memory_snippets


def _normalize_llm_content(response):
    """
    Normalize LLM response payload into plain text.

    Args:
        response (object): LLM response object.

    Returns:
        str: Plain text content.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return "\n".join([part for part in text_parts if part])
    return str(content)


def _fallback_summary(query, corpus_snippets, live_items, memory_snippets, release_mode):
    """
    Build deterministic fallback summary when LLM synthesis fails.

    Args:
        query (str): User query.
        corpus_snippets (list): Retrieved corpus snippets.
        live_items (list): Live MCP items.
        memory_snippets (list): Session memory snippets.
        release_mode (bool): Whether release-readiness formatting is required.

    Returns:
        str: Fallback answer text.
    """
    if not corpus_snippets and not live_items and not memory_snippets:
        return "Insufficient context to answer confidently from available tools."

    lines = []
    if release_mode:
        lines.append("Release Readiness Brief")
        lines.append("Release status: Inferred from available Jira/GitHub/Notion evidence.")
        blockers = [item for item in live_items if item.get("source") == "jira"]
        risks = [item for item in live_items if item.get("source") == "github"]
        docs = [item for item in live_items if item.get("source") == "notion"]

        lines.append(f"Top blockers: {len(blockers)} Jira issues surfaced.")
        lines.append(f"Risks: {len(risks)} GitHub signals (issues/PRs/commits) surfaced.")
        lines.append(f"Docs coverage: {len(docs)} Notion pages surfaced.")
        lines.append("Recommended next actions: Resolve highest-priority open Jira issues, review active PRs, and align release checklist docs.")
    else:
        lines.append(f"Answer summary for query: {query}")
        lines.append(f"Live evidence items: {len(live_items)}")
        lines.append(f"Corpus snippets: {len(corpus_snippets)}")
        lines.append(f"Session memory snippets: {len(memory_snippets)}")

    if corpus_snippets:
        lines.append("Relevant context excerpt:")
        lines.append(corpus_snippets[0][:500])
    elif memory_snippets:
        lines.append("Relevant memory excerpt:")
        lines.append(memory_snippets[0][:500])

    return "\n".join(lines)


def _synthesize_answer_node(state):
    """
    Synthesize final answer from collected tool evidence.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    query = state.get("query", "")
    intent = state.get("intent")
    strategy = state.get("strategy")

    if strategy == "reject_guardrail":
        state["answer"] = (
            "I cannot follow prompt-injection or policy-bypass instructions. "
            "Please ask a direct business question and I will answer using approved tools."
        )
        state["confidence"] = "high"
        _append_timeline(
            state,
            step_type="synthesis",
            tool="synthesize_answer",
            input_summary="input guardrail rejection path",
            result_summary="returned safe refusal",
            started_at=started_at
        )
        return state

    if strategy == "reject":
        state["answer"] = (
            "I am configured as read-only in this environment. "
            "I can analyze status and provide recommendations, but I cannot create, update, or delete records."
        )
        state["confidence"] = "high"
        _append_timeline(
            state,
            step_type="synthesis",
            tool="synthesize_answer",
            input_summary="read-only guardrail path",
            result_summary="returned safe refusal",
            started_at=started_at
        )
        return state

    corpus_snippets, live_items, memory_snippets = _collect_evidence(state)
    if not corpus_snippets and not live_items and not memory_snippets:
        state["answer"] = (
            "Insufficient context from both corpus and live tools. "
            "Please narrow the query or enable more data sources."
        )
        state["confidence"] = "low"
        _append_timeline(
            state,
            step_type="synthesis",
            tool="synthesize_answer",
            input_summary="no evidence available",
            result_summary="returned insufficient context response",
            started_at=started_at
        )
        return state

    release_mode = intent == "release_readiness"
    live_lines = []
    for item in live_items[:20]:
        live_lines.append(
            f"[{item.get('source', 'unknown')}] "
            f"title={item.get('title', '')} | status={item.get('status', '')} | "
            f"priority={item.get('priority', '')} | assignee={item.get('assignee', '')} | "
            f"url={item.get('url', '')}"
        )

    corpus_lines = [snippet[:700] for snippet in corpus_snippets[:6]]
    memory_lines = [snippet[:500] for snippet in memory_snippets[:4]]
    synthesis_prompt = ""
    if release_mode:
        synthesis_prompt = f"""
You are a release-readiness analyst.
Use only the evidence below and do not invent facts.
If evidence is weak, state uncertainty clearly.
If session memory conflicts with live evidence, prioritize live evidence.

User query:
{query}

Live tool evidence:
{chr(10).join(live_lines) if live_lines else "No live evidence"}

Corpus evidence:
{chr(10).join(corpus_lines) if corpus_lines else "No corpus evidence"}

Session memory evidence:
{chr(10).join(memory_lines) if memory_lines else "No memory evidence"}

Return exactly these sections:
1) Release status
2) Top blockers
3) Risks
4) Recommended next actions
5) Evidence links summary
"""
    else:
        synthesis_prompt = f"""
Answer the user query strictly from the evidence below.
Do not hallucinate or infer beyond evidence.
If uncertain, explicitly say so.
If session memory conflicts with live evidence, prioritize live evidence.

User query:
{query}

Live tool evidence:
{chr(10).join(live_lines) if live_lines else "No live evidence"}

Corpus evidence:
{chr(10).join(corpus_lines) if corpus_lines else "No corpus evidence"}

Session memory evidence:
{chr(10).join(memory_lines) if memory_lines else "No memory evidence"}
"""

    answer = ""
    try:
        llm = get_active_llm()
        response = llm.invoke(synthesis_prompt)
        answer = _normalize_llm_content(response).strip()
    except Exception:
        answer = _fallback_summary(query, corpus_snippets, live_items, memory_snippets, release_mode)

    if not answer:
        answer = _fallback_summary(query, corpus_snippets, live_items, memory_snippets, release_mode)

    has_live = len(live_items) > 0
    has_corpus = len(corpus_snippets) > 0
    has_memory = len(memory_snippets) > 0
    if has_live and has_corpus:
        confidence = "high"
    elif has_live or has_corpus or has_memory:
        confidence = "medium"
    else:
        confidence = "low"

    state["answer"] = answer
    state["confidence"] = confidence
    _append_timeline(
        state,
        step_type="synthesis",
        tool="synthesize_answer",
        input_summary=f"intent={intent}, strategy={strategy}",
        result_summary=f"answer_length={len(answer)}, confidence={confidence}",
        started_at=started_at
    )
    return state


def _cite_sources_node(state):
    """
    Collect and deduplicate citations from all tool outputs.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    citations = []
    for _, payload in state["tool_outputs"].items():
        for link in payload.get("raw_source_links", []):
            if link:
                citations.append(link)
        preferred_urls = payload.get("metadata_urls")
        if preferred_urls is None:
            preferred_urls = payload.get("urls", [])
        for link in preferred_urls:
            if link:
                citations.append(link)

    deduped = []
    seen = set()
    for citation in citations:
        if citation in seen:
            continue
        seen.add(citation)
        deduped.append(citation)

    state["citations"] = deduped[:20]
    _append_timeline(
        state,
        step_type="plan",
        tool="cite_sources",
        input_summary=f"tool_outputs={len(state['tool_outputs'])}",
        result_summary=f"citations={len(state['citations'])}",
        started_at=started_at
    )
    return state


def _output_guardrail_node(state):
    """
    Apply output citation gate for freshness-sensitive responses.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    gate = check_output_citation_gate(
        intent=state.get("intent", ""),
        query_text=state.get("query", ""),
        answer_text=state.get("answer", ""),
        citations=state.get("citations", [])
    )
    state["citation_gate_passed"] = gate.get("passed", True)
    if not gate.get("passed", True):
        state["guardrail_flags"].append("citation_gate")
        state["answer"] = (
            "I cannot provide a freshness-sensitive answer without verifiable citations. "
            "Please enable live sources or refine the query."
        )
        state["confidence"] = "low"

    _append_timeline(
        state,
        step_type="guardrail",
        tool="output_citation_gate",
        input_summary=f"intent={state.get('intent', '')}, citations={len(state.get('citations', []))}",
        result_summary=f"passed={gate.get('passed', True)} reason={gate.get('reason', '')}",
        started_at=started_at,
        status="ok" if gate.get("passed", True) else "warning"
    )
    return state


def _persist_memory_node(state):
    """
    Persist sanitized memory candidate when memory is enabled.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Updated runtime state.
    """
    started_at = time.time()
    if not state.get("memory_enabled"):
        _append_timeline(
            state,
            step_type="memory",
            tool="write_memory",
            input_summary="memory disabled",
            result_summary="memory write skipped",
            started_at=started_at,
            status="skipped"
        )
        return state

    memory_candidate = extract_memory_candidate(
        query_text=state.get("query", ""),
        answer_text=state.get("answer", "")
    )
    if not memory_candidate:
        _append_timeline(
            state,
            step_type="memory",
            tool="write_memory",
            input_summary="no memory candidate",
            result_summary="no write",
            started_at=started_at,
            status="skipped"
        )
        return state

    sanitized = sanitize_memory_text(memory_candidate)
    if not sanitized.get("safe"):
        state["guardrail_flags"].append("memory_safety")
        _append_timeline(
            state,
            step_type="guardrail",
            tool="memory_safety_filter",
            input_summary=memory_candidate[:120],
            result_summary=f"blocked reason={sanitized.get('reason', '')}",
            started_at=started_at,
            status="warning"
        )
        return state

    memory_write_result = _run_tool(
        state,
        "write_memory",
        {
            "memory_text": sanitized.get("text", ""),
            "session_id": state.get("session_id", "default"),
            "tags": ["user_preference"],
            "ttl_days": state.get("memory_ttl_days", 30)
        }
    )
    state["memory_written"] = bool(memory_write_result.get("written"))
    if memory_write_result.get("written"):
        state["memory_write_id"] = memory_write_result.get("memory_id", "")
    _append_timeline(
        state,
        step_type="memory",
        tool="persist_memory",
        input_summary=f"candidate={memory_candidate[:80]}",
        result_summary=f"written={state['memory_written']}",
        started_at=started_at
    )
    return state


def _run_manual_flow(state):
    """
    Execute agent flow in deterministic order without LangGraph.

    Args:
        state (dict): Runtime state.

    Returns:
        dict: Final runtime state.
    """
    state = _classify_intent_node(state)
    state = _input_guardrail_node(state)
    state = _retrieve_memory_node(state)
    state = _decide_strategy_node(state)
    state = _retrieve_or_tool_call_node(state)
    state = _synthesize_answer_node(state)
    state = _cite_sources_node(state)
    state = _output_guardrail_node(state)
    state = _persist_memory_node(state)
    return state


def _build_langgraph():
    """
    Build LangGraph runtime graph for agent orchestration.

    Returns:
        object: Compiled graph.
    """
    graph = StateGraph(dict)
    graph.add_node("classify_intent", _classify_intent_node)
    graph.add_node("input_guardrail", _input_guardrail_node)
    graph.add_node("retrieve_memory", _retrieve_memory_node)
    graph.add_node("decide_strategy", _decide_strategy_node)
    graph.add_node("retrieve_or_tool_call", _retrieve_or_tool_call_node)
    graph.add_node("synthesize_answer", _synthesize_answer_node)
    graph.add_node("cite_sources", _cite_sources_node)
    graph.add_node("output_guardrail", _output_guardrail_node)
    graph.add_node("persist_memory", _persist_memory_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "input_guardrail")
    graph.add_edge("input_guardrail", "retrieve_memory")
    graph.add_edge("retrieve_memory", "decide_strategy")
    graph.add_edge("decide_strategy", "retrieve_or_tool_call")
    graph.add_edge("retrieve_or_tool_call", "synthesize_answer")
    graph.add_edge("synthesize_answer", "cite_sources")
    graph.add_edge("cite_sources", "output_guardrail")
    graph.add_edge("output_guardrail", "persist_memory")
    graph.add_edge("persist_memory", END)
    return graph.compile()


def run_agent(
    query,
    enabled_sources,
    model_name,
    session_id="default",
    memory_enabled=False,
    memory_top_k=3
):
    """
    Run the hybrid agentic workflow and return structured output.

    Args:
        query (str): User query.
        enabled_sources (list): Source filters selected in the UI.
        model_name (str): Selected model name.
        session_id (str, optional): Session identifier for memory.
        memory_enabled (bool, optional): Enable session memory.
        memory_top_k (int, optional): Top memory hits.

    Returns:
        dict: Agent response payload with answer, citations, timeline, mode, confidence.
    """
    _ensure_model(model_name)
    selected_sources = _normalize_enabled_sources(enabled_sources)
    tools = get_agent_tools(
        selected_sources,
        session_id=session_id,
        memory_enabled=memory_enabled,
        memory_top_k=memory_top_k
    )
    tool_map = {tool["name"]: tool for tool in tools}

    state = {
        "query": query,
        "enabled_sources": selected_sources,
        "model_name": model_name,
        "session_id": session_id,
        "memory_enabled": bool(memory_enabled),
        "memory_top_k": int(memory_top_k),
        "memory_ttl_days": 30,
        "intent": "",
        "strategy": "",
        "answer": "",
        "confidence": "low",
        "timeline": [],
        "tool_outputs": {},
        "tool_errors": {},
        "circuit_breaker": {},
        "tool_map": tool_map,
        "citations": [],
        "memory_snippets": [],
        "memory_items": [],
        "memory_hit": False,
        "memory_written": False,
        "memory_write_id": "",
        "input_guardrail": {},
        "citation_gate_passed": True,
        "guardrail_flags": [],
        "force_safe_response": False
    }

    if LANGGRAPH_AVAILABLE:
        try:
            graph = _build_langgraph()
            state = graph.invoke(state)
            mode_used = "agentic_langgraph"
        except Exception:
            state = _run_manual_flow(state)
            mode_used = "agentic_fallback"
    else:
        state = _run_manual_flow(state)
        mode_used = "agentic_fallback"

    return {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "timeline": state.get("timeline", []),
        "mode_used": mode_used,
        "confidence": state.get("confidence", "low"),
        "memory_hit": state.get("memory_hit", False),
        "memory_written": state.get("memory_written", False),
        "citation_gate_passed": state.get("citation_gate_passed", True),
        "guardrail_triggered": len(state.get("guardrail_flags", [])) > 0,
        "guardrail_flags": state.get("guardrail_flags", [])
    }
