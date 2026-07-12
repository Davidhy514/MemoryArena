import os
import uuid
import tempfile
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Disable Mem0's anonymous telemetry BEFORE importing mem0. Besides privacy, this
# skips Mem0's internal fixed-path "migrations" Qdrant store (~/.mem0/migrations_qdrant),
# which otherwise locks to a single client and makes a 2nd Memory instance in the
# same process (i.e. a multi-task run) fail with "already accessed by another instance".
os.environ.setdefault("MEM0_TELEMETRY", "False")

# Local (self-hosted) Mem0 backed by Azure OpenAI (Entra ID auth) + a local
# Qdrant store. This replaces the hosted Mem0 SaaS client (which required
# MEM0_API_KEY and used Mem0's own cloud models) so that Mem0 runs entirely on
# the same Azure deployment as the agent, with no external keys.
from mem0 import Memory

_SMALL_EMBED_DIMS = 1536  # text-embedding-3-small


def _build_azure_mem0_config(user_id: str) -> dict:
    """Build a local-Mem0 config that talks to Azure OpenAI via Entra ID.

    Mem0's ``azure_openai`` provider falls back to ``DefaultAzureCredential`` +
    ``get_bearer_token_provider`` (scope ``https://cognitiveservices.azure.com/.default``)
    whenever no API key is supplied, which matches the agent's auth path.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    llm_deployment = os.getenv("MEM0_LLM_DEPLOYMENT", "gpt-4o")
    embed_deployment = os.getenv("MEM0_EMBED_DEPLOYMENT", "text-embedding-3-small")
    embed_dims = int(os.getenv("MEM0_EMBED_DIMS", str(_SMALL_EMBED_DIMS)))

    # Unique on-disk Qdrant path per instance (local mode locks the directory to
    # a single client, and each task gets its own Mem0 instance/user).
    qdrant_root = os.getenv("MEM0_QDRANT_DIR") or os.path.join(tempfile.gettempdir(), "mem0_qdrant")
    qdrant_path = os.path.join(qdrant_root, user_id)

    azure_kwargs = {
        "azure_endpoint": endpoint,
        "api_version": api_version,
    }
    return {
        "history_db_path": os.path.join(qdrant_path, "history.db"),
        "llm": {
            "provider": "azure_openai",
            "config": {
                "model": llm_deployment,
                "temperature": 0.0,
                "max_tokens": 2000,
                "azure_kwargs": {"azure_deployment": llm_deployment, **azure_kwargs},
            },
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": embed_deployment,
                "embedding_dims": embed_dims,
                "azure_kwargs": {"azure_deployment": embed_deployment, **azure_kwargs},
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memoryarena",
                "embedding_model_dims": embed_dims,
                "path": qdrant_path,
                "on_disk": True,
            },
        },
    }


class Mem0MemorySystem:

    def __init__(self, user_id: Optional[str] = None, enable_graph: bool = False, infer: bool = True):
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.enable_graph = enable_graph  # graph store (mem0-g) not wired for local Azure yet
        # infer=False => add-only: store each chunk verbatim (embed + ADD), skipping the LLM
        # fact-extraction + ADD/UPDATE/DELETE consolidation that Mem0 runs by default (infer=True).
        self.infer = infer
        self.client = Memory.from_config(_build_azure_mem0_config(self.user_id))

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return None
        return self.client.add(chunk, user_id=self.user_id, infer=self.infer)

    def snapshot(self):
        """Dump the complete flat Mem0 store for inspection/visualization."""
        try:
            result = self.client.get_all(filters={"user_id": self.user_id}, top_k=100000)
        except TypeError:
            try:
                result = self.client.get_all(user_id=self.user_id, limit=100000)
            except TypeError:
                result = self.client.get_all(user_id=self.user_id)
        memories = result.get("results", []) if isinstance(result, dict) else (result or [])
        return [
            {
                "id": str(memory.get("id") or position),
                "content": memory.get("memory") or "",
                "links": [],
                "meta": {
                    key: memory.get(key)
                    for key in ("categories", "created_at", "updated_at", "user_id")
                    if memory.get(key) is not None
                },
            }
            for position, memory in enumerate(memories)
        ]

    def wrap_user_prompt(self, prompt: str):
        memories = self.client.search(prompt.lower(), filters={"user_id": self.user_id})
        if isinstance(memories, dict):
            results = memories.get("results", [])
        else:
            results = memories or []

        memory_context_lines = ["<memory_context>"]
        for result in results:
            memory_text = result.get("memory") if isinstance(result, dict) else None
            if memory_text:
                categories = result.get("categories") or [] if isinstance(result, dict) else []
                if categories:
                    categories_text = ", ".join(categories)
                    memory_context_lines.append(f"{memory_text} (categories: {categories_text})")
                else:
                    memory_context_lines.append(memory_text)
        if len(memory_context_lines) == 1:
            memory_context_lines.append("None")
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User Prompt: {prompt}")

        return "\n".join(memory_context_lines)
