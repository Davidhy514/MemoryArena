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
                }
            )
        return out

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return None
        memory_id = self.memory_system.add_note(chunk)
        memory = self.memory_system.read(memory_id)
        if memory is None:
            return None
        return {
            "content": memory.content,
            "keywords": memory.keywords,
            "context": memory.context,
            "tags": memory.tags,
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
