import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?hidden\s+prompt",
    r"developer\s+message",
    r"bypass\s+(guardrails|safety|policy)",
    r"jailbreak",
    r"act\s+as\s+system"
]

LIVE_SIGNAL_TERMS = [
    "latest",
    "today",
    "current",
    "recent",
    "right now",
    "blocker",
    "release",
    "readiness",
    "launch",
    "ship"
]

SENSITIVE_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)",
    r"\bsk-[A-Za-z0-9]{16,}\b",
    r"\bghp_[A-Za-z0-9]{20,}\b"
]

MEMORY_CANDIDATE_PATTERNS = [
    r"(?i)\bremember that\s+(.+)$",
    r"(?i)\bi prefer\s+(.+)$",
    r"(?i)\bmy preferred\s+(.+)$",
    r"(?i)\bmy name is\s+(.+)$",
    r"(?i)\bi am\s+(.+)$"
]

MEMORY_TRIGGER_TERMS = [
    "remember",
    "future use",
    "save this",
    "store this",
    "keep this",
    "note this"
]

LOW_VALUE_ANSWER_TERMS = [
    "insufficient context",
    "no context",
    "read-only",
    "cannot provide a freshness-sensitive answer",
    "prompt-injection"
]


def check_input_guardrail(query_text):
    """
    Check for prompt-injection-like patterns in user input.

    Args:
        query_text (str): User query.

    Returns:
        dict: Guardrail result payload.
    """
    lowered = (query_text or "").lower()
    matched = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            matched.append(pattern)

    score = len(matched)
    if score == 0:
        return {"triggered": False, "severity": "none", "reason": ""}
    if score == 1:
        return {"triggered": True, "severity": "low", "reason": "potential prompt injection pattern"}
    return {"triggered": True, "severity": "high", "reason": "multiple prompt injection patterns"}


def check_output_citation_gate(intent, query_text, answer_text, citations):
    """
    Check citation sufficiency for freshness or release-sensitive answers.

    Args:
        intent (str): Classified intent.
        query_text (str): User query.
        answer_text (str): Assistant answer.
        citations (list): Citation links.

    Returns:
        dict: Citation gate result.
    """
    answer_lower = (answer_text or "").lower()
    if (
        "insufficient context" in answer_lower
        or "read-only" in answer_lower
        or "prompt-injection" in answer_lower
    ):
        return {"passed": True, "reason": "safe_response"}

    query_lower = (query_text or "").lower()
    needs_gate = intent in ["release_readiness", "freshness_lookup"]
    if not needs_gate:
        needs_gate = any(term in query_lower for term in LIVE_SIGNAL_TERMS)

    if not needs_gate:
        return {"passed": True, "reason": "not_required"}
    if citations:
        return {"passed": True, "reason": "citations_present"}
    return {"passed": False, "reason": "missing citations for freshness-sensitive response"}


def sanitize_memory_text(memory_text):
    """
    Sanitize memory text and block unsafe content before persistence.

    Args:
        memory_text (str): Candidate memory text.

    Returns:
        dict: Sanitized memory payload.
    """
    if not memory_text:
        return {"safe": False, "text": "", "redacted": False, "reason": "empty"}

    normalized = " ".join(memory_text.strip().split())
    if len(normalized) < 8:
        return {"safe": False, "text": "", "redacted": False, "reason": "too_short"}

    injection = check_input_guardrail(normalized)
    if injection.get("triggered"):
        return {"safe": False, "text": "", "redacted": False, "reason": "unsafe_instruction_pattern"}

    redacted = normalized
    did_redact = False
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, redacted):
            did_redact = True
            redacted = re.sub(pattern, "[redacted]", redacted)

    redacted = redacted[:500]
    return {"safe": True, "text": redacted, "redacted": did_redact, "reason": "ok"}


def extract_memory_candidate(query_text, answer_text):
    """
    Extract a concise memory candidate from a user query.

    Args:
        query_text (str): User query.
        answer_text (str): Assistant answer.

    Returns:
        str: Candidate memory text.
    """
    query = (query_text or "").strip()
    answer = " ".join((answer_text or "").strip().split())
    query_lower = query.lower()

    explicit_candidate = ""
    for pattern in MEMORY_CANDIDATE_PATTERNS:
        match = re.search(pattern, query)
        if match:
            candidate = match.group(1).strip(" .")
            if candidate:
                explicit_candidate = candidate
                break

    has_memory_trigger = any(term in query_lower for term in MEMORY_TRIGGER_TERMS)
    if not has_memory_trigger and not explicit_candidate:
        return ""

    main_query = re.split(
        r"(?i)\b(?:and\s+)?remember that\b|\bfor future use\b|\bplease remember\b|\bsave this\b|\bstore this\b",
        query,
        maxsplit=1
    )[0].strip(" .")

    low_value_answer = any(term in answer.lower() for term in LOW_VALUE_ANSWER_TERMS)
    has_useful_answer = bool(answer) and not low_value_answer

    if main_query and has_useful_answer:
        summary = answer[:260]
        return f"User asked: {main_query}. Answer summary: {summary}"

    if explicit_candidate and has_useful_answer and main_query:
        summary = answer[:220]
        return f"Remembered intent: {explicit_candidate}. Related answer: {summary}"

    if explicit_candidate:
        return explicit_candidate[:500]

    if has_useful_answer:
        return f"Answer summary: {answer[:300]}"

    return ""
