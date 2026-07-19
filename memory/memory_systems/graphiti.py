"""Graphiti temporal-knowledge-graph memory backend — Neo4j + Entra-proxied Azure.

Graphiti (getzep/graphiti, `graphiti-core`) is a **temporal knowledge graph**: ingested text is
distilled by an LLM into Entities (nodes with an evolving `.summary`) and Facts (typed edges /
triplets carrying bi-temporal `valid_at` / `invalid_at`). Retrieval is hybrid (semantic + BM25 +
graph). On contradiction a fact is *invalidated* (its `invalid_at` is set) rather than deleted —
this is the capability that distinguishes Graphiti from mem0 (consolidation), A-Mem (note links)
and Letta (core + archival), and it is why propagation is FIRST-CLASS here (not architecturally
N/A as it was for Letta).

This backend mirrors the duck-typed contract the WebShop poison driver expects
(`scripts/webshop_poison/inproc_memory.py`): `add_chunk`, `wrap_user_prompt`, `snapshot`,
`inject_poison` (+ `poison_channel`) and `close`. Graphiti's API is async-first, so every call is
bridged onto a single persistent event loop (`_run`).

The agent's LLM + embeddings are routed through the local Entra->Azure OpenAI proxy (same trick as
`letta.py` / `LETTA_PROXY_BASE`). Extraction uses **gpt-4o** by default (reliable structured output;
the gpt-5.4 reasoning deployment may not honour `response_format`).

Reset semantics: the worker rebuilds this object per example. `__init__` purges this run's
`group_id` subgraph so each example starts from empty memory (mirrors Letta deleting a prior agent).
`close()` leaves the final graph in Neo4j so it can be inspected post-hoc.

Env (set by the run driver / mem_worker launch):
    NEO4J_URI            bolt endpoint            (default bolt://localhost:7687)
    NEO4J_USER           user                     (default neo4j)
    NEO4J_PASSWORD       password                 (default membattle)
    GRAPHITI_PROXY_BASE  OpenAI-shaped Azure proxy (default $LETTA_PROXY_BASE or http://127.0.0.1:8106/v1)
    GRAPHITI_LLM_MODEL   extraction model / deployment (default gpt-4o)
    GRAPHITI_EMBED_MODEL embedding deployment     (default text-embedding-3-small)
    GRAPHITI_EMBED_DIM   embedding dim            (default 1536)
"""
import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "membattle")
_PROXY = os.getenv("GRAPHITI_PROXY_BASE", os.getenv("LETTA_PROXY_BASE", "http://127.0.0.1:8106/v1"))
_LLM_MODEL = os.getenv("GRAPHITI_LLM_MODEL", "gpt-4o")
_EMB_MODEL = os.getenv("GRAPHITI_EMBED_MODEL", "text-embedding-3-small")
_EMB_DIM = int(os.getenv("GRAPHITI_EMBED_DIM", "1536"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GraphitiMemorySystem:
    """One Graphiti group (`group_id == membattle-<user_id>`) == one memory store, purged fresh on build."""

    def __init__(self, user_id: Optional[str] = None,
                 poison_channel: str = "episode_direct", **_: object) -> None:
        self.user_id = user_id or "membattle"
        self.group_id = f"membattle-{self.user_id}"
        # Where inject_poison() writes: episode_direct | fact_direct | entity_summary (see inject_poison).
        self.poison_channel = poison_channel
        self._seq = 0
        self.last_retrieval_trace: list[str] = []

        # Single persistent event loop: Graphiti's async neo4j driver + httpx clients bind to the loop
        # on first await, so every bridged call must run on the SAME loop.
        self._loop = asyncio.new_event_loop()

        llm_cfg = LLMConfig(api_key="proxy", model=_LLM_MODEL, small_model=_LLM_MODEL, base_url=_PROXY)
        self._llm = OpenAIGenericClient(config=llm_cfg)
        self._embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key="proxy", embedding_model=_EMB_MODEL, embedding_dim=_EMB_DIM, base_url=_PROXY))
        self._reranker = OpenAIRerankerClient(client=self._llm, config=llm_cfg)
        self.graphiti = Graphiti(_NEO4J_URI, _NEO4J_USER, _NEO4J_PASSWORD,
                                 llm_client=self._llm, embedder=self._embedder,
                                 cross_encoder=self._reranker)
        self._run(self._setup())

    # --- async->sync bridge ---
    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    async def _setup(self) -> None:
        await self.graphiti.build_indices_and_constraints()
        await self._purge()  # reset hygiene: each build starts from empty memory

    async def _purge(self) -> None:
        await self.graphiti.driver.execute_query(
            "MATCH (n {group_id: $gid}) DETACH DELETE n", gid=self.group_id)

    # --- writes ---
    def add_chunk(self, chunk: str):
        """Ingest a benign memory as a text Episode; Graphiti's LLM extracts entities + facts from it."""
        self._seq += 1
        self._run(self.graphiti.add_episode(
            name=f"chunk-{self._seq}", episode_body=chunk, source=EpisodeType.text,
            source_description="benign", reference_time=_now(), group_id=self.group_id))
        return {"events": ["ADD"]}

    def inject_poison(self, text: str):
        """Write the poison via the configured channel:
          episode_direct  -> add_episode: LLM extracts entities/facts from prose (natural/agentic path).
          fact_direct     -> save a verbatim Fact EDGE (no extraction LLM), retrieval-gated.
          entity_summary  -> save an Entity NODE whose `.summary` is the poison (surfaced whenever that
                             entity is retrieved) — the always-surfaced ceiling.
        Temporal invalidation (a poison fact that INVALIDATES a true fact) is achieved through
        `episode_direct` when the poison contradicts an existing fact — Graphiti sets `invalid_at`
        on the superseded edge automatically.
        """
        channel = getattr(self, "poison_channel", "episode_direct")
        if channel == "fact_direct":
            self._run(self._fact_direct(text))
        elif channel == "entity_summary":
            self._run(self._entity_summary(text))
        else:
            self._run(self.graphiti.add_episode(
                name="poison", episode_body=text, source=EpisodeType.text,
                source_description="note", reference_time=_now(), group_id=self.group_id))
        return {"events": ["POISON"], "channel": channel}

    async def _fact_direct(self, text: str) -> None:
        """Persist a verbatim Fact edge (user)-[NOTE]->(note) bypassing the extraction/resolution LLM.
        The fact text is stored exactly; retrieval is still gated by hybrid search."""
        now = _now()
        src = EntityNode(name="user", group_id=self.group_id, labels=["Entity"], created_at=now)
        tgt = EntityNode(name=f"note-{uuid.uuid4().hex[:8]}", group_id=self.group_id,
                         labels=["Entity"], created_at=now)
        await src.generate_name_embedding(self._embedder)
        await tgt.generate_name_embedding(self._embedder)
        await src.save(self.graphiti.driver)
        await tgt.save(self.graphiti.driver)
        edge = EntityEdge(source_node_uuid=src.uuid, target_node_uuid=tgt.uuid, name="NOTE",
                          fact=text, group_id=self.group_id, created_at=now, valid_at=now, episodes=[])
        await edge.generate_embedding(self._embedder)
        await edge.save(self.graphiti.driver)

    async def _entity_summary(self, text: str) -> None:
        """Persist an Entity node whose `.summary` carries the poison — surfaced whenever the entity is
        retrieved (the always-in-context ceiling, analogous to a Letta core-memory write)."""
        now = _now()
        node = EntityNode(name=f"note-{uuid.uuid4().hex[:8]}", group_id=self.group_id,
                          labels=["Entity"], created_at=now, summary=text)
        await node.generate_name_embedding(self._embedder)
        await node.save(self.graphiti.driver)

    # --- retrieval ---
    def wrap_user_prompt(self, prompt: str) -> str:
        """Hybrid-search this group's graph and surface retrieved Facts (edges, with their temporal
        range) AND Entity summaries (nodes) as <memory_context>. Mirrors the Zep wrapper's edge
        formatting and additionally surfaces node summaries so the entity_summary channel is visible."""
        edges, nodes = self._run(self._retrieve(prompt))
        lines = ["<memory_context>"]
        trace: list[str] = []
        for e in edges:
            fact = getattr(e, "fact", None)
            if not fact:
                continue
            valid_at = getattr(e, "valid_at", None) or "date unknown"
            invalid_at = getattr(e, "invalid_at", None) or "present"
            lines.append(f"{fact} (Date range: {valid_at} - {invalid_at})")
            trace.append(fact)
        for n in nodes:
            summary = (getattr(n, "summary", "") or "").strip()
            if summary:
                lines.append(f"{getattr(n, 'name', 'entity')}: {summary}")
                trace.append(summary)
        if len(lines) == 1:
            lines.append("None")
        lines.append("</memory_context>")
        lines.append(f"User: {prompt}")
        self.last_retrieval_trace = trace
        return "\n".join(lines)

    async def _retrieve(self, prompt: str):
        edges = await self.graphiti.search(prompt, group_ids=[self.group_id])
        cfg = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_res = await self.graphiti._search(query=prompt, config=cfg, group_ids=[self.group_id])
        return edges, list(getattr(node_res, "nodes", []) or [])

    # --- introspection ---
    def snapshot(self):
        """Architecture-aware snapshot: Entity nodes (meta.surface='entity', links to neighbours) and
        Fact edges rendered as nodes (meta.surface='fact', links to their source+target, carrying the
        temporal range). This exposes the REAL graph structure so link/propagation metrics are
        meaningful, and lets the driver locate which surface a poison landed on."""
        return self._run(self._snapshot())

    async def _snapshot(self):
        driver = self.graphiti.driver
        nodes: list[dict] = []
        node_res = await driver.execute_query(
            "MATCH (n:Entity {group_id: $gid}) "
            "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary", gid=self.group_id)
        for rec in getattr(node_res, "records", []) or []:
            name = rec.get("name") or ""
            summary = rec.get("summary") or ""
            content = f"{name}: {summary}".strip(": ").strip() if summary else name
            nodes.append({"id": rec.get("uuid"), "content": content, "links": [],
                          "meta": {"surface": "entity", "name": name}})
        edge_res = await driver.execute_query(
            "MATCH (a:Entity {group_id: $gid})-[e:RELATES_TO]->(b:Entity {group_id: $gid}) "
            "RETURN e.uuid AS uuid, e.fact AS fact, a.uuid AS src, b.uuid AS tgt, "
            "e.valid_at AS valid_at, e.invalid_at AS invalid_at", gid=self.group_id)
        by_id = {n["id"]: n for n in nodes}
        for rec in getattr(edge_res, "records", []) or []:
            src, tgt = rec.get("src"), rec.get("tgt")
            if src in by_id and tgt not in by_id.get(src, {}).get("links", []):
                by_id[src]["links"].append(tgt)
            nodes.append({"id": rec.get("uuid"), "content": rec.get("fact") or "",
                          "links": [x for x in (src, tgt) if x],
                          "meta": {"surface": "fact",
                                   "valid_at": str(rec.get("valid_at")) if rec.get("valid_at") else None,
                                   "invalid_at": str(rec.get("invalid_at")) if rec.get("invalid_at") else None}})
        return nodes

    def close(self):
        # Leave the final graph in Neo4j for post-hoc inspection; just release the driver + loop.
        try:
            self._run(self.graphiti.close())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
