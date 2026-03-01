from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import OpenAIEmbeddings

load_dotenv()

MEMORY_DIR = Path("memory_data")
MEMORY_INDEX_DIR = MEMORY_DIR / "index"
MEMORY_RECORDS_FILE = MEMORY_DIR / "memory_records.jsonl"
DEFAULT_TTL_DAYS = 30

_memory_embeddings = None


class HashEmbeddings:
    """
    Deterministic local embeddings fallback when OpenAI embeddings are unavailable.

    Returns:
        list: Stable fixed-size vectors.
    """

    def __init__(self, dimension=256):
        self.dimension = dimension

    def _embed_text(self, text):
        raw = (text or "").encode("utf-8", errors="ignore")
        digest = sha256(raw).digest()
        values = []
        while len(values) < self.dimension:
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) >= self.dimension:
                    break
            digest = sha256(digest).digest()
        return values

    def embed_documents(self, texts):
        """
        Embed a list of texts.

        Args:
            texts (list): Text documents.

        Returns:
            list: Vector embeddings.
        """
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        """
        Embed a query string.

        Args:
            text (str): Query text.

        Returns:
            list: Vector embedding.
        """
        return self._embed_text(text)

    def __call__(self, text):
        """
        Compatibility hook for FAISS versions expecting callable embeddings.

        Args:
            text (str): Input text.

        Returns:
            list: Vector embedding.
        """
        return self._embed_text(text)


def _utc_now_iso():
    """
    Return current UTC timestamp in ISO format.

    Returns:
        str: UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value):
    """
    Parse an ISO timestamp into a timezone-aware datetime.

    Args:
        value (str): ISO timestamp.

    Returns:
        datetime: Parsed datetime value.
    """
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _record_expired(record):
    """
    Check if a memory record has expired.

    Args:
        record (dict): Memory record payload.

    Returns:
        bool: True when expired.
    """
    expires_at = _parse_iso(record.get("expires_at", ""))
    if expires_at is None:
        return False
    return expires_at <= datetime.now(timezone.utc)


def _ensure_memory_dir():
    """
    Ensure memory directories and sidecar files exist.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_RECORDS_FILE.exists():
        MEMORY_RECORDS_FILE.touch()


def _get_memory_embeddings():
    """
    Return embeddings implementation for memory retrieval.

    Returns:
        object: Embeddings client.
    """
    global _memory_embeddings
    if _memory_embeddings is not None:
        return _memory_embeddings

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            _memory_embeddings = OpenAIEmbeddings()
            return _memory_embeddings
        except Exception:
            pass

    _memory_embeddings = HashEmbeddings()
    return _memory_embeddings


def _default_placeholder_record():
    """
    Build placeholder metadata for an empty FAISS index.

    Returns:
        dict: Placeholder record metadata.
    """
    return {
        "memory_id": "__placeholder__",
        "session_id": "__system__",
        "source": "memory",
        "created_at": _utc_now_iso(),
        "expires_at": "",
        "is_placeholder": True,
        "tags": []
    }


def _build_faiss_from_records(records):
    """
    Build FAISS index from active records with one placeholder document.

    Args:
        records (list): Memory record payloads.

    Returns:
        FAISS: Vector store instance.
    """
    docs = [Document(page_content="memory placeholder", metadata=_default_placeholder_record())]
    for record in records:
        docs.append(
            Document(
                page_content=record.get("text", ""),
                metadata={
                    "memory_id": record.get("memory_id", ""),
                    "session_id": record.get("session_id", ""),
                    "source": "memory",
                    "created_at": record.get("created_at", ""),
                    "expires_at": record.get("expires_at", ""),
                    "is_placeholder": False,
                    "tags": record.get("tags", [])
                }
            )
        )
    return FAISS.from_documents(docs, _get_memory_embeddings())


def _save_memory_store(store):
    """
    Persist FAISS memory store locally.

    Args:
        store (FAISS): Vector store.
    """
    store.save_local(str(MEMORY_INDEX_DIR))


def _load_memory_store():
    """
    Load FAISS memory store from disk.

    Returns:
        FAISS: Loaded vector store.
    """
    return FAISS.load_local(
        str(MEMORY_INDEX_DIR),
        _get_memory_embeddings(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE
    )


def _load_all_records():
    """
    Load all memory records from JSONL sidecar.

    Returns:
        list: List of memory records.
    """
    _ensure_memory_dir()
    records = []
    with open(MEMORY_RECORDS_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _write_all_records(records):
    """
    Overwrite sidecar with provided records.

    Args:
        records (list): Memory record list.
    """
    _ensure_memory_dir()
    with open(MEMORY_RECORDS_FILE, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _active_records(records):
    """
    Return non-expired memory records.

    Args:
        records (list): Record payloads.

    Returns:
        list: Active records.
    """
    return [record for record in records if not _record_expired(record)]


def _normalize_for_dedupe(text):
    """
    Normalize text for duplicate checks.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    return " ".join((text or "").strip().lower().split())


def init_memory_index():
    """
    Initialize memory index and sidecar store if missing.

    Returns:
        dict: Initialization status.
    """
    _ensure_memory_dir()
    if (MEMORY_INDEX_DIR / "index.faiss").exists() and (MEMORY_INDEX_DIR / "index.pkl").exists():
        return {"status": "ready", "initialized": False}

    records = _active_records(_load_all_records())
    store = _build_faiss_from_records(records)
    _save_memory_store(store)
    return {"status": "ready", "initialized": True}


def search_memory(query, session_id, top_k=3):
    """
    Search session memory with TTL-aware filtering.

    Args:
        query (str): Query text.
        session_id (str): Session identifier.
        top_k (int, optional): Number of records to return.

    Returns:
        dict: Normalized memory retrieval payload.
    """
    init_memory_index()
    safe_top_k = max(1, int(top_k))

    all_records = _load_all_records()
    active_records = _active_records(all_records)
    active_by_id = {record.get("memory_id", ""): record for record in active_records if record.get("memory_id")}
    if not active_by_id:
        return {
            "items": [],
            "snippets": [],
            "memory_ids": [],
            "raw_source_links": [],
            "freshness_ts": _utc_now_iso(),
            "retrieval_score": None
        }

    store = _load_memory_store()
    docs_with_scores = []
    try:
        docs_with_scores = store.similarity_search_with_score(
            query,
            k=safe_top_k * 4,
            filter={"session_id": session_id}
        )
    except Exception:
        docs = store.similarity_search(query, k=safe_top_k * 4)
        docs_with_scores = [(doc, None) for doc in docs]

    items = []
    snippets = []
    score_values = []
    seen = set()
    for doc, score in docs_with_scores:
        metadata = doc.metadata or {}
        if metadata.get("is_placeholder"):
            continue
        memory_id = metadata.get("memory_id", "")
        if not memory_id or memory_id in seen:
            continue
        record = active_by_id.get(memory_id)
        if not record:
            continue
        if record.get("session_id", "") != session_id:
            continue

        seen.add(memory_id)
        text = record.get("text", "")
        snippets.append(text[:1200])
        items.append({
            "id": memory_id,
            "title": "Session memory",
            "summary": text[:500],
            "status": "active",
            "priority": "",
            "assignee": "",
            "source": "memory",
            "type": "memory",
            "repo": "",
            "url": "",
            "updated_at": record.get("created_at", "")
        })
        if score is not None:
            score_values.append(float(score))
        if len(items) >= safe_top_k:
            break

    retrieval_score = None
    if score_values:
        retrieval_score = sum(score_values) / len(score_values)

    return {
        "items": items,
        "snippets": snippets,
        "memory_ids": [item["id"] for item in items],
        "raw_source_links": [],
        "freshness_ts": _utc_now_iso(),
        "retrieval_score": retrieval_score
    }


def write_memory(memory_text, session_id, tags=None, ttl_days=DEFAULT_TTL_DAYS):
    """
    Write one memory record if not duplicate.

    Args:
        memory_text (str): Memory content.
        session_id (str): Session identifier.
        tags (list, optional): Tags for the record.
        ttl_days (int, optional): Time-to-live in days.

    Returns:
        dict: Write result payload.
    """
    init_memory_index()
    text_value = (memory_text or "").strip()
    if not text_value:
        return {"written": False, "reason": "empty_memory"}

    all_records = _load_all_records()
    active_session_records = [
        record for record in _active_records(all_records)
        if record.get("session_id", "") == session_id
    ]

    normalized = _normalize_for_dedupe(text_value)
    for record in active_session_records:
        if _normalize_for_dedupe(record.get("text", "")) == normalized:
            return {"written": False, "reason": "duplicate_memory", "memory_id": record.get("memory_id", "")}

    ttl_value = max(1, int(ttl_days))
    created_at = datetime.now(timezone.utc)
    record = {
        "memory_id": str(uuid4()),
        "session_id": session_id,
        "text": text_value,
        "tags": tags or [],
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(days=ttl_value)).isoformat(),
        "source": "memory",
        "pii_redacted": False
    }

    with open(MEMORY_RECORDS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    store = _load_memory_store()
    store.add_documents(
        [
            Document(
                page_content=text_value,
                metadata={
                    "memory_id": record["memory_id"],
                    "session_id": session_id,
                    "source": "memory",
                    "created_at": record["created_at"],
                    "expires_at": record["expires_at"],
                    "is_placeholder": False,
                    "tags": record["tags"]
                }
            )
        ]
    )
    _save_memory_store(store)

    return {"written": True, "memory_id": record["memory_id"], "expires_at": record["expires_at"]}


def clear_memory(session_id):
    """
    Clear all memory records for a session and rebuild index.

    Args:
        session_id (str): Session identifier.

    Returns:
        dict: Clear result payload.
    """
    init_memory_index()
    all_records = _load_all_records()
    remaining_records = []
    cleared_count = 0
    for record in all_records:
        if record.get("session_id", "") == session_id:
            cleared_count += 1
            continue
        remaining_records.append(record)

    _write_all_records(remaining_records)
    active_remaining = _active_records(remaining_records)
    rebuilt_store = _build_faiss_from_records(active_remaining)
    _save_memory_store(rebuilt_store)
    return {"cleared_count": cleared_count}
