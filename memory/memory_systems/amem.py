import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from agentic_memory.memory_system import AgenticMemorySystem
except ImportError:
    print("AgenticMemorySystem not found, please install AMEM")


class AMemMemorySystem:

    def __init__(
        self,
        user_id: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.user_id = user_id
        # Route A-Mem's metadata/evolution LLM to Azure OpenAI (Entra ID) when
        # AZURE_OPENAI_ENDPOINT is set, mirroring the other backends. Embeddings
        # stay local (sentence-transformers). Otherwise fall back to OpenAI.
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            llm_model = os.getenv("AMEM_LLM_DEPLOYMENT", "gpt-4o")
            api_key = api_key or "azure-entra-placeholder"  # satisfies the OpenAI controller ctor
        elif api_key is None and llm_backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        self.memory_system = AgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
        )
        if azure_endpoint:
            self._route_llm_to_azure(azure_endpoint, llm_model)
        self.last_retrieval_trace = []
        self.last_write_trace = {}

    @staticmethod
    def _note_state(note):
        return (note.context, tuple(note.tags or []), tuple(note.links or []))

    def _normalize_and_sync_notes(self, before: dict) -> dict:
        """Keep links valid and push evolution's neighbor metadata edits into Chroma.

        A-Mem mutates neighbor context/tags in `self.memories`, but the original implementation
        leaves the Chroma metadata stale. That makes intended contamination invisible when a
        neighbor is retrieved directly. Re-index only notes that actually changed.
        """
        ms = self.memory_system
        valid_ids = set(ms.memories)
        changed = []
        removed = 0
        for memory_id, note in list(ms.memories.items()):
            links = []
            for link in note.links or []:
                link_id = str(link)
                if link_id in valid_ids and link_id != memory_id and link_id not in links:
                    links.append(link_id)
                else:
                    removed += 1
            note.links = links
            if before.get(memory_id) != self._note_state(note):
                changed.append(memory_id)
        for memory_id in changed:
            note = ms.memories[memory_id]
            ms.update(memory_id, context=note.context, tags=note.tags, links=note.links)
        return {"updated_note_ids": changed, "invalid_links_removed": removed}

    def _route_llm_to_azure(self, azure_endpoint: str, deployment: str) -> None:
        from openai import AzureOpenAI
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_ad_token_provider=token_provider,
        )
        # A-Mem's OpenAI controller calls client.chat.completions.create(model=self.model, ...);
        # an AzureOpenAI client with the deployment name is drop-in compatible.
        self.memory_system.llm_controller.llm.client = client
        self.memory_system.llm_controller.llm.model = deployment

    def snapshot(self):
        """Dump the agentic store: notes with their links + evolving context/tags."""
        notes = getattr(self.memory_system, "memories", {}) or {}
        out = []
        for mid, n in notes.items():
            out.append(
                {
                    "id": str(mid),
                    "content": getattr(n, "content", ""),
                    "links": [str(x) for x in (getattr(n, "links", None) or [])],
                    "context": getattr(n, "context", ""),
                    "tags": list(getattr(n, "tags", None) or []),
                    "keywords": list(getattr(n, "keywords", None) or []),
                    "meta": {
                        "context": getattr(n, "context", ""),
                        "tags": list(getattr(n, "tags", None) or []),
                        "keywords": list(getattr(n, "keywords", None) or []),
                        "category": getattr(n, "category", "Uncategorized"),
                        "timestamp": getattr(n, "timestamp", ""),
                        "retrieval_count": getattr(n, "retrieval_count", 0),
                    },
                }
            )
        return out

    def save_state(self, path):
        """Serialize the full A-Mem state (note/link graph + evolution counter) to a JSON file.

        The Chroma vector index is NOT saved: it is a pure function of the notes and is rebuilt
        with local embeddings (no LLM) in load_state(). This lets a base memory reload in ~1s
        instead of re-running the per-note metadata + link-evolution LLM calls.
        """
        import json as _json

        ms = self.memory_system
        notes = getattr(ms, "memories", {}) or {}
        payload = {
            "version": 1,
            "model_name": getattr(ms, "model_name", "all-MiniLM-L6-v2"),
            "evo_cnt": int(getattr(ms, "evo_cnt", 0) or 0),
            "notes": [
                {
                    "id": str(n.id),
                    "content": n.content,
                    "keywords": list(n.keywords or []),
                    "links": list(n.links or []),
                    "context": n.context,
                    "category": getattr(n, "category", "Uncategorized"),
                    "tags": list(n.tags or []),
                    "timestamp": n.timestamp,
                    "last_accessed": n.last_accessed,
                    "retrieval_count": int(n.retrieval_count or 0),
                    "evolution_history": list(n.evolution_history or []),
                }
                for n in notes.values()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False)
        return {"n_notes": len(payload["notes"]),
                "n_links": sum(len(n["links"]) for n in payload["notes"])}

    def load_state(self, path):
        """Rebuild A-Mem from a saved state file: reconstruct the note/link graph, then re-embed
        into a fresh Chroma collection (local embeddings, NO LLM). Inverse of save_state()."""
        import json as _json
        from agentic_memory.memory_system import MemoryNote

        with open(path, encoding="utf-8") as f:
            payload = _json.load(f)
        ms = self.memory_system
        ms.memories = {}
        for nd in payload.get("notes", []):
            note = MemoryNote(
                content=nd.get("content", ""),
                id=nd.get("id"),
                keywords=nd.get("keywords") or [],
                links=nd.get("links") or [],
                retrieval_count=nd.get("retrieval_count") or 0,
                timestamp=nd.get("timestamp"),
                last_accessed=nd.get("last_accessed"),
                context=nd.get("context") or "General",
                evolution_history=nd.get("evolution_history") or [],
                category=nd.get("category") or "Uncategorized",
                tags=nd.get("tags") or [],
            )
            ms.memories[note.id] = note
        # Saved research states may contain LLM-hallucinated/dangling link ids. They cannot be
        # traversed, so drop them before rebuilding the derived vector index.
        valid_ids = set(ms.memories)
        removed = 0
        for memory_id, note in ms.memories.items():
            clean = []
            for link in note.links or []:
                link_id = str(link)
                if link_id in valid_ids and link_id != memory_id and link_id not in clean:
                    clean.append(link_id)
                else:
                    removed += 1
            note.links = clean
        ms.evo_cnt = int(payload.get("evo_cnt", 0) or 0)
        # Rebuild the Chroma vector index from the notes -- local embeddings, NO LLM calls.
        ms.consolidate_memories()
        return {"n_notes": len(ms.memories), "invalid_links_removed": removed}

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return None
        before = {
            memory_id: self._note_state(note)
            for memory_id, note in self.memory_system.memories.items()
        }
        memory_id = self.memory_system.add_note(chunk)
        sync = self._normalize_and_sync_notes(before)
        memory = self.memory_system.read(memory_id)
        if memory is None:
            return None
        self.last_write_trace = {
            "memory_id": memory_id,
            "links": list(memory.links or []),
            **sync,
        }
        return {
            "content": memory.content,
            "keywords": memory.keywords,
            "context": memory.context,
            "tags": memory.tags,
            "trace": self.last_write_trace,
        }

    def wrap_user_prompt(self, prompt: str):
        # search_agentic() is A-Mem's canonical retrieval: vector kNN over the
        # store, THEN expansion along each hit's links -- the memory->memory
        # channel that flat/linear stores do not have. Link-pulled notes are
        # flagged is_neighbor and prefixed "(via link)" so the hop is visible.
        # AMEM_SEARCH_K lets experiments tighten the retrieval budget (default 5)
        # so a poison just outside top-k can ONLY arrive via a link.
        try:
            _k = int(os.environ.get("AMEM_SEARCH_K", "5"))
        except (TypeError, ValueError):
            _k = 5
        results = self.memory_system.search_agentic(prompt.lower(), k=_k)
        expand_links = os.environ.get("AMEM_LINK_EXPANSION", "1").strip().lower() \
            not in {"0", "false", "no", "off"}
        if not expand_links:
            results = [result for result in results if not result.get("is_neighbor")]
        self.last_retrieval_trace = [
            {
                "id": str(result.get("id", "")),
                "is_neighbor": bool(result.get("is_neighbor")),
                "score": result.get("score"),
                "content": (result.get("content") or "")[:1200],
                "context": result.get("context") or "",
                "tags": list(result.get("tags") or []),
            }
            for result in results
        ]
        memory_context_lines = ["<memory_context>"]

        if not results:
            memory_context_lines.append("None")
        else:
            for result in results:
                memory_text = result.get("content")
                if not memory_text:
                    continue
                tags = result.get("tags") or []
                keywords = result.get("keywords") or []
                context = result.get("context") or ""

                meta_parts = []
                if tags:
                    meta_parts.append(f"tags: {', '.join(tags)}")
                if keywords:
                    meta_parts.append(f"keywords: {', '.join(keywords)}")
                if context:
                    meta_parts.append(f"context: {context}")

                prefix = "(via link) " if result.get("is_neighbor") else ""
                if meta_parts:
                    memory_context_lines.append(f"{prefix}{memory_text} ({'; '.join(meta_parts)})")
                else:
                    memory_context_lines.append(f"{prefix}{memory_text}")

            if len(memory_context_lines) == 1:
                memory_context_lines.append("None")

        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User Prompt: {prompt}")

        return "\n".join(memory_context_lines)
