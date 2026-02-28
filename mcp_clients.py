from dotenv import load_dotenv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from github import Github
from jira import JIRA
from notion_client import Client
import os
import time

load_dotenv()

READ_ONLY_TOOL_ALLOWLIST = {
    "github": {"read_release_state"},
    "jira": {"read_release_state"},
    "notion": {"read_release_state"}
}

_github_client = None
_jira_client = None
_notion_client = None


def _utc_now_iso():
    """
    Return current UTC timestamp in ISO-8601 format.

    Returns:
        str: Current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _extract_terms(query):
    """
    Extract simple lowercase terms from a query string.

    Args:
        query (str): User query.

    Returns:
        list: Tokenized terms.
    """
    if not query:
        return []
    return [term.strip().lower() for term in query.replace(",", " ").split() if term.strip()]


def _execute_with_retry(func, args, timeout_seconds=12, retries=2):
    """
    Execute a function with timeout and retry policy.

    Args:
        func (callable): Function to execute.
        args (dict): Function arguments.
        timeout_seconds (int, optional): Timeout in seconds.
        retries (int, optional): Retry count.

    Returns:
        dict: Function result payload.
    """
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, args)
                return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            if attempt > retries:
                raise TimeoutError(f"Tool call timed out after {timeout_seconds}s") from exc
            time.sleep(0.3 * attempt)
        except Exception:
            if attempt > retries:
                raise
            time.sleep(0.3 * attempt)
    return {"items": [], "raw_source_links": [], "freshness_ts": _utc_now_iso()}


def _get_github_client():
    """
    Return authenticated GitHub client.

    Returns:
        Github: Authenticated GitHub client.
    """
    global _github_client
    if _github_client is None:
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not token:
            raise ValueError("GITHUB_ACCESS_TOKEN is not configured")
        _github_client = Github(token)
    return _github_client


def _get_jira_client():
    """
    Return authenticated Jira client.

    Returns:
        JIRA: Authenticated Jira client.
    """
    global _jira_client
    if _jira_client is None:
        username = os.getenv("JIRA_USERNAME")
        token = os.getenv("JIRA_API_TOKEN")
        instance_url = os.getenv("JIRA_INSTANCE_URL")
        if not username or not token or not instance_url:
            raise ValueError("JIRA_USERNAME, JIRA_API_TOKEN, and JIRA_INSTANCE_URL must be configured")
        _jira_client = JIRA(server=instance_url, basic_auth=(username, token))
    return _jira_client


def _get_notion_client():
    """
    Return authenticated Notion client.

    Returns:
        Client: Authenticated Notion client.
    """
    global _notion_client
    if _notion_client is None:
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY is not configured")
        _notion_client = Client(auth=api_key)
    return _notion_client


def _matches_terms(text, terms):
    """
    Check whether any search terms exist in text.

    Args:
        text (str): Input text.
        terms (list): Search terms.

    Returns:
        bool: True if a match exists or no terms were provided.
    """
    if not terms:
        return True
    lowered = (text or "").lower()
    return any(term in lowered for term in terms)


def _github_read_release_state(args):
    """
    Read release-related information from GitHub.

    Args:
        args (dict): Tool arguments.

    Returns:
        dict: Normalized tool result.
    """
    github_client = _get_github_client()
    query = args.get("query", "")
    terms = _extract_terms(query)
    repo_limit = int(args.get("repo_limit", 3))
    item_limit = int(args.get("item_limit", 5))

    items = []
    links = []
    added = 0

    repos = github_client.get_user().get_repos(sort="updated")
    for repo in repos:
        if repo_limit <= 0 or added >= item_limit:
            break
        repo_limit -= 1

        for issue in repo.get_issues(state="open", sort="updated", direction="desc"):
            if added >= item_limit:
                break
            issue_text = f"{issue.title} {issue.body or ''}"
            if not _matches_terms(issue_text, terms):
                continue
            item_type = "pull_request" if issue.pull_request else "issue"
            label_names = [label.name for label in issue.labels] if issue.labels else []
            items.append({
                "id": str(issue.number),
                "title": issue.title,
                "summary": issue.body or "",
                "status": issue.state,
                "priority": ",".join(label_names),
                "assignee": issue.user.login if issue.user else "unknown",
                "source": "github",
                "type": item_type,
                "repo": repo.full_name,
                "url": issue.html_url,
                "updated_at": issue.updated_at.isoformat() if issue.updated_at else ""
            })
            links.append(issue.html_url)
            added += 1

        if added >= item_limit:
            continue

        commit_count = 0
        for commit in repo.get_commits():
            if added >= item_limit or commit_count >= 2:
                break
            message = commit.commit.message or ""
            if not _matches_terms(message, terms):
                commit_count += 1
                continue
            items.append({
                "id": commit.sha[:8],
                "title": f"Commit in {repo.full_name}",
                "summary": message,
                "status": "recent_commit",
                "priority": "",
                "assignee": commit.commit.author.name if commit.commit.author else "unknown",
                "source": "github",
                "type": "commit",
                "repo": repo.full_name,
                "url": commit.html_url,
                "updated_at": commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else ""
            })
            links.append(commit.html_url)
            added += 1
            commit_count += 1

    return {
        "items": items,
        "raw_source_links": list(dict.fromkeys(links)),
        "freshness_ts": _utc_now_iso()
    }


def _jira_read_release_state(args):
    """
    Read release-related issue information from Jira.

    Args:
        args (dict): Tool arguments.

    Returns:
        dict: Normalized tool result.
    """
    jira_client = _get_jira_client()
    query = args.get("query", "")
    terms = _extract_terms(query)
    max_results = int(args.get("item_limit", 8))

    jql = "statusCategory != Done ORDER BY updated DESC"
    issues = jira_client.search_issues(jql, maxResults=max_results * 3)

    items = []
    links = []
    instance_url = os.getenv("JIRA_INSTANCE_URL", "").rstrip("/")

    for issue in issues:
        summary = issue.fields.summary if issue.fields.summary else ""
        description = str(issue.fields.description) if issue.fields.description else ""
        combined = f"{summary} {description}"
        if not _matches_terms(combined, terms):
            continue
        assignee = issue.fields.assignee.displayName if issue.fields.assignee else "unassigned"
        priority = issue.fields.priority.name if issue.fields.priority else "unknown"
        status = issue.fields.status.name if issue.fields.status else "unknown"
        issue_url = f"{instance_url}/browse/{issue.key}" if instance_url else ""

        items.append({
            "id": issue.key,
            "title": summary,
            "summary": description[:1000],
            "status": status,
            "priority": priority,
            "assignee": assignee,
            "source": "jira",
            "type": "issue",
            "repo": "",
            "url": issue_url,
            "updated_at": issue.fields.updated if issue.fields.updated else ""
        })
        if issue_url:
            links.append(issue_url)
        if len(items) >= max_results:
            break

    return {
        "items": items,
        "raw_source_links": list(dict.fromkeys(links)),
        "freshness_ts": _utc_now_iso()
    }


def _extract_notion_title(page):
    """
    Extract human-readable page title from Notion page payload.

    Args:
        page (dict): Notion page payload.

    Returns:
        str: Parsed title.
    """
    properties = page.get("properties", {})
    for _, value in properties.items():
        if value.get("type") == "title":
            title_data = value.get("title", [])
            if title_data:
                text = "".join([part.get("plain_text", "") for part in title_data]).strip()
                if text:
                    return text
    return page.get("id", "untitled")


def _notion_read_release_state(args):
    """
    Read release-related pages from Notion.

    Args:
        args (dict): Tool arguments.

    Returns:
        dict: Normalized tool result.
    """
    notion_client = _get_notion_client()
    query = args.get("query", "")
    max_results = int(args.get("item_limit", 6))

    search_payload = notion_client.search(
        query=query,
        filter={"property": "object", "value": "page"},
        page_size=max_results
    )

    items = []
    links = []
    for page in search_payload.get("results", []):
        title = _extract_notion_title(page)
        page_url = page.get("url", "")
        items.append({
            "id": page.get("id", ""),
            "title": title,
            "summary": f"Notion page: {title}",
            "status": "available",
            "priority": "",
            "assignee": "",
            "source": "notion",
            "type": "page",
            "repo": "",
            "url": page_url,
            "updated_at": page.get("last_edited_time", "")
        })
        if page_url:
            links.append(page_url)

    return {
        "items": items,
        "raw_source_links": list(dict.fromkeys(links)),
        "freshness_ts": _utc_now_iso()
    }


def run_mcp_tool(server, tool, args):
    """
    Run a read-only MCP-style tool and return normalized result payload.

    Args:
        server (str): Tool server namespace (github, jira, notion).
        tool (str): Tool name to run.
        args (dict): Tool arguments.

    Returns:
        dict: Normalized tool output.
    """
    if server not in READ_ONLY_TOOL_ALLOWLIST:
        raise ValueError(f"Unsupported server: {server}")
    if tool not in READ_ONLY_TOOL_ALLOWLIST[server]:
        raise PermissionError(f"Tool {tool} is not permitted for server {server}")

    if server == "github":
        return _execute_with_retry(_github_read_release_state, args)
    if server == "jira":
        return _execute_with_retry(_jira_read_release_state, args)
    if server == "notion":
        return _execute_with_retry(_notion_read_release_state, args)
    raise ValueError(f"No tool dispatcher configured for server {server}")
