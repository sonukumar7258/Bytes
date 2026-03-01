# Import necessary libraries
import re
import threading
from urllib.parse import unquote
import solara
import reacton.ipyvuetify as v

from agent_runtime import run_agent
from memory_store import clear_memory
from ai_assistant_chain import (
    answer_from_context_fallback,
    extract_urls,
    load_pdf_vector,
    load_vector_store,
    switch_llm
)


# Initialize reactive variables
notion_checkbox = solara.reactive(True)
jira_checkbox = solara.reactive(False)
github_checkbox = solara.reactive(False)
website_checkbox = solara.reactive(False)
file_upload_checkbox = solara.reactive(False)
llm = solara.reactive("llama3-70b-8192")
assistant_mode = solara.reactive("rag")
show_timeline = solara.reactive(True)
memory_enabled = solara.reactive(False)
session_id = solara.reactive("default")
memory_top_k = solara.reactive(3)


def get_selected_sources():
    """
    Get source filters selected in the sidebar.

    Returns:
        list: Selected source names.
    """
    source_list = []
    if notion_checkbox.value:
        source_list.append("notion")
    if jira_checkbox.value:
        source_list.append("jira")
    if github_checkbox.value:
        source_list.append("github")
    if website_checkbox.value:
        source_list.append("website")
    return source_list


def _tokenize_text(text):
    """
    Tokenize text into lowercase alphanumeric keywords.

    Args:
        text (str): Input text.

    Returns:
        set: Token set.
    """
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _url_keywords(url):
    """
    Extract searchable keywords from URL.

    Args:
        url (str): URL value.

    Returns:
        set: URL keyword set.
    """
    if not url:
        return set()
    decoded = unquote(url).lower()
    cleaned = decoded.replace("https://", " ").replace("http://", " ")
    cleaned = cleaned.replace("/", " ").replace("-", " ").replace("_", " ").replace(".", " ")
    # Remove raw UUID-like blocks from scoring.
    cleaned = re.sub(r"\b[0-9a-f]{32}\b", " ", cleaned)
    return _tokenize_text(cleaned)


def _score_url(url, query_text, answer_text):
    """
    Score URL relevance against query and answer text.

    Args:
        url (str): URL value.
        query_text (str): User query text.
        answer_text (str): Assistant answer text.

    Returns:
        int: Relevance score.
    """
    query_tokens = _tokenize_text(query_text)
    answer_tokens = _tokenize_text(answer_text)
    url_tokens = _url_keywords(url)

    if not url_tokens:
        return 0

    score = 0
    score += len(url_tokens.intersection(query_tokens)) * 4
    score += len(url_tokens.intersection(answer_tokens)) * 2

    # Penalize low-signal internal root-style Notion links.
    if "notion.so/bytecorp/" in url and len(url_tokens.intersection(query_tokens.union(answer_tokens))) == 0:
        score -= 2

    return score


def format_related_links(urls, query_text="", answer_text="", max_links=2):
    """
    Format links into markdown with dedupe and relevance ranking.

    Args:
        urls (list): URL list.
        query_text (str, optional): User query text.
        answer_text (str, optional): Assistant answer text.
        max_links (int, optional): Maximum number of links to display.

    Returns:
        str: Markdown-safe HTML output.
    """
    notion_by_id = {}
    other_urls = []

    for url in urls:
        if "notion" in url:
            match = re.search(r"([0-9a-f]{32})", url.replace("-", ""))
            if match:
                uid = match.group(1)
                existing = notion_by_id.get(uid, "")
                if len(url) > len(existing):
                    notion_by_id[uid] = url
            else:
                other_urls.append(url)
        else:
            other_urls.append(url)

    deduped_urls = list(notion_by_id.values()) + other_urls
    if max_links and len(deduped_urls) > max_links:
        scored_urls = []
        for index, url in enumerate(deduped_urls):
            score = _score_url(url, query_text, answer_text)
            scored_urls.append((url, score, index))

        positive_scored = [item for item in scored_urls if item[1] > 0]
        if len(positive_scored) >= max_links:
            scored_urls = positive_scored

        scored_urls.sort(key=lambda item: (-item[1], item[2]))
        top_scored = scored_urls[:max_links]
        top_scored.sort(key=lambda item: item[2])
        deduped_urls = [item[0] for item in top_scored]

    if not deduped_urls:
        return ""

    link_markdown = "Related Links: <br />"
    for url in deduped_urls:
        link_markdown += f"[{url}]({url})<br />"
    return link_markdown


def format_timeline_markdown(timeline):
    """
    Render timeline payload into concise markdown.

    Args:
        timeline (list): List of timeline step dictionaries.

    Returns:
        str: Markdown content.
    """
    if not timeline:
        return ""

    lines = ["### Tool Timeline"]
    step_number = 1
    for step in timeline:
        step_type = step.get("step_type", "step")
        tool = step.get("tool", "unknown")
        status = step.get("status", "ok")
        duration = step.get("duration_ms", 0)
        input_summary = step.get("input_summary", "")
        result_summary = step.get("result_summary", "")
        lines.append(
            f"**{step_number}. {step_type.upper()}** `{tool}` ({status}, {duration} ms)<br />"
            f"Input: {input_summary}<br />Result: {result_summary}<br />"
        )
        step_number += 1
    return "\n".join(lines)


# Load LLM with default set to llama3-70b-8192 and generate retrieval chain using it, using notion by default
switch_llm("llama", "llama-3.3-70b-versatile")
global retrieval_chain_body
retrieval_chain_body = load_vector_store("all", ["notion"])


def on_value_change_tools(value, name):
    """
    Callback function to handle changes in checkbox values for different tools.

    Args:
        value (bool): The new value of the checkbox.
        name (str): The name of the tool associated with the checkbox.
    """
    global retrieval_chain_body
    if value:
        file_upload_checkbox.value = False
    retrieval_chain_body = load_vector_store("all", get_selected_sources())


def on_value_change_llm(value):
    """
    Callback function to handle changes in the LLM selection.

    Args:
        value (str): The new model name.
    """
    global retrieval_chain_body
    if value == "gemini":
        switch_llm("gemini", "gemini-pro")
    else:
        if value == "qwen/qwen3-32b":
            switch_llm("llama", "qwen/qwen3-32b")
        elif value == "llama3-8b-8192":
            switch_llm("llama", "llama-3.1-8b-instant")
        elif value == "llama3-70b-8192":
            switch_llm("llama", "llama-3.3-70b-versatile")
        elif value == "moonshotai/kimi-k2-instruct":
            switch_llm("llama", "moonshotai/kimi-k2-instruct")
        elif value == "GPT5.2":
            switch_llm("gpt", "gpt-5.2")
    retrieval_chain_body = load_vector_store("all", get_selected_sources())


def on_value_change_file(value):
    """
    Callback function to handle changes in the file upload checkbox.

    Args:
        value (bool): The new value of the file upload checkbox.
    """
    global retrieval_chain_body
    if value:
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False
        assistant_mode.value = "rag"
    else:
        retrieval_chain_body = load_vector_store("all", get_selected_sources())


def on_session_id_change(value):
    """
    Callback for session id updates used by memory mode.

    Args:
        value (str): New session id.
    """
    cleaned = (value or "").strip()
    session_id.value = cleaned if cleaned else "default"


image_url = "logo2.png"


@solara.component
def Page():
    """
    Solara component for the AI Assistant page.
    """
    llms = [
        "gemini",
        "qwen/qwen3-32b",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "moonshotai/kimi-k2-instruct",
        "GPT5.2"
    ]

    solara.Title("AI Assistant")
    solara.Image(image_url, width="300px")

    loader, set_loader = solara.use_state(False)
    input_message, set_input_message = solara.use_state("")
    output_message, set_output_message = solara.use_state("")
    output_urls, set_output_urls = solara.use_state("")
    output_timeline, set_output_timeline = solara.use_state("")
    processing, set_processing = solara.use_state(False)
    memory_status, set_memory_status = solara.use_state("")

    def query_assistant_body(user_input):
        """
        Query either RAG mode or Agentic mode and update UI state.

        Args:
            user_input (str): User query text.
        """
        set_processing(True)
        try:
            cleaned_input = (user_input or "").strip()
            if not cleaned_input:
                set_output_message("Please enter a question.")
                set_output_urls("")
                set_output_timeline("")
                return

            global retrieval_chain_body
            selected_sources = get_selected_sources()

            if assistant_mode.value == "agentic" and not file_upload_checkbox.value:
                agent_response = run_agent(
                    query=cleaned_input,
                    enabled_sources=selected_sources,
                    model_name=llm.value,
                    session_id=session_id.value,
                    memory_enabled=memory_enabled.value,
                    memory_top_k=memory_top_k.value
                )
                answer = agent_response.get("answer", "")
                confidence = agent_response.get("confidence", "low")
                mode_used = agent_response.get("mode_used", "agentic")
                citations = agent_response.get("citations", [])
                timeline = agent_response.get("timeline", [])
                memory_hit = agent_response.get("memory_hit", False)
                memory_written = agent_response.get("memory_written", False)
                guardrail_triggered = agent_response.get("guardrail_triggered", False)
                citation_gate_passed = agent_response.get("citation_gate_passed", True)

                answer_with_meta = (
                    f"{answer}\n\nConfidence: {confidence}\nMode: {mode_used}"
                    f"\nMemory hit: {memory_hit}\nMemory written: {memory_written}"
                    f"\nCitation gate passed: {citation_gate_passed}\nGuardrail triggered: {guardrail_triggered}"
                )
                set_output_message(answer_with_meta.replace("\n", "<br />"))
                set_output_urls(format_related_links(citations, cleaned_input, answer, max_links=2))

                if show_timeline.value:
                    set_output_timeline(format_timeline_markdown(timeline))
                else:
                    set_output_timeline("")
                return

            # RAG path
            response = retrieval_chain_body.invoke({"input": cleaned_input})
            context = response.get("context", [])
            context_text = ""
            url_source_text = ""

            if context:
                if not file_upload_checkbox.value:
                    first_source = context[0].metadata.get("source", "")
                    if first_source == "notion":
                        context_text = context[0].page_content
                        if len(context) > 1:
                            context_text += "\n" + context[1].page_content
                        url_source_text = context[0].page_content
                    else:
                        for document in context:
                            if document.metadata.get("source") != "notion":
                                context_text += document.page_content
                        url_source_text = context_text
                else:
                    for document in context:
                        context_text += document.page_content
                    url_source_text = context_text

                with open("context.txt", "w", encoding="utf-8") as context_file:
                    context_file.write(context_text)

            urls = extract_urls(url_source_text)

            answer = response.get("answer", "no context")
            if answer == "no context":
                fallback_answer = answer_from_context_fallback(cleaned_input, context_text)
                if fallback_answer and fallback_answer.lower() != "no context":
                    answer = fallback_answer
                elif len(urls) > 0:
                    answer = "I do not have the exact knowledge for what you asked, but these links may help."
                else:
                    answer = "Sorry, I can not help with this query from the current context."

            set_output_message(answer.replace("\n", "<br />"))
            set_output_urls(format_related_links(urls, cleaned_input, answer, max_links=2))
            set_output_timeline("")
        finally:
            set_loader(False)
            set_processing(False)

    def handle_update(*ignore_args):
        """
        Handle Enter key or Send click.
        """
        if not processing:
            set_loader(True)
            thread = threading.Thread(target=query_assistant_body, args=(input_message,))
            thread.start()

    def on_file(file_info):
        """
        Handle file upload and load PDF retrieval chain.

        Args:
            file_info (FileInfo): Uploaded file details.
        """
        global retrieval_chain_body
        set_filename(file_info["name"])
        set_size(file_info["size"])
        content = file_info["file_obj"].read(file_info["size"])
        set_content(content)

        file_path = file_info["name"]
        with open(file_path, "wb") as file_handle:
            file_handle.write(content)
        retrieval_chain_body = load_pdf_vector(file_path)

    def clear_memory_for_session():
        """
        Clear memory records for active session id.
        """
        active_session = (session_id.value or "default").strip() or "default"
        result = clear_memory(active_session)
        set_memory_status(
            f"Cleared {result.get('cleared_count', 0)} memory entries for session `{active_session}`."
        )

    content, set_content = solara.use_state(b"")
    filename, set_filename = solara.use_state("")
    size, set_size = solara.use_state(0)

    with solara.Column() as main:
        with solara.Sidebar():
            solara.Select(label="Select Model", value=llm, values=llms, on_value=on_value_change_llm)
            solara.Select(label="Assistant Mode", value=assistant_mode, values=["rag", "agentic"])
            solara.Switch(label="Show Tool Timeline", value=show_timeline)
            solara.Switch(label="Enable Memory", value=memory_enabled)
            solara.Select(label="Memory Top K", value=memory_top_k, values=[1, 2, 3, 4, 5])
            session_id_field = v.TextField(
                v_model=session_id.value,
                label="Session ID",
                on_v_model=on_session_id_change
            )
            solara.Button(label="Clear Memory", on_click=clear_memory_for_session)

            solara.Switch(
                label="Notion",
                value=notion_checkbox,
                on_value=lambda value: on_value_change_tools(value, "notion")
            )
            solara.Switch(
                label="Jira",
                value=jira_checkbox,
                on_value=lambda value: on_value_change_tools(value, "jira")
            )
            solara.Switch(
                label="Github",
                value=github_checkbox,
                on_value=lambda value: on_value_change_tools(value, "github")
            )
            solara.Switch(
                label="Website",
                value=website_checkbox,
                on_value=lambda value: on_value_change_tools(value, "website")
            )
            solara.Switch(
                label="File Upload",
                value=file_upload_checkbox,
                on_value=lambda value: on_value_change_file(value)
            )

        with solara.Card(title="Instructions"):
            solara.Markdown(
                """ **Use Agentic mode for release-readiness and latest-status questions.**
**Use RAG mode for strict corpus-grounded Q&A or PDF uploads.**<br />
Enable memory only when you want session personalization and continuity.<br />
Example (Agentic): "Give me latest blockers for release X with Jira, GitHub and Notion evidence."<br />
Example (Memory): "Remember that my preferred release timezone is PST."<br />
Example (RAG): "Summarize what we documented about onboarding from Notion docs."<br />
The assistant remains read-only for external systems.
"""
            )

        text_field = v.TextField(v_model=input_message, on_v_model=set_input_message, label="Enter your message")
        v.use_event(text_field, "keydown.enter", handle_update)
        solara.Button(label="Send", color="primary", on_click=handle_update)
        solara.ProgressLinear(loader)

        if file_upload_checkbox.value:
            solara.FileDrop(label="Drag and drop a pdf.", on_file=on_file, lazy=True)

        solara.Markdown(output_message)
        solara.Markdown(output_urls)
        if memory_status:
            solara.Markdown(memory_status.replace("\n", "<br />"))
        if show_timeline.value:
            solara.Markdown(output_timeline)
