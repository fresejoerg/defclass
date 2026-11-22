"""Hierarchical two-stage (Cluster -> Leaf) routing for large label spaces."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import re
from typing import Any

from .providers import LLMProvider
from .types import ClassDefinition, ClusterSummarySchema

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ClusterNode:
    """Single super-class cluster with summary text and member labels."""

    cluster_id: str
    summary_name: str
    summary_text: str
    members: list[str]


def _embed_definitions(texts: list[str], embed_model: str | None) -> list[list[float]]:
    """Embed definitions with SentenceTransformer if installed, else normalized TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embed_model or "sentence-transformers/all-MiniLM-L6-v2")
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    except ImportError:
        pass

    # Lightweight zero-dependency TF-IDF fallback
    docs = [_TOKEN_RE.findall(t.lower()) for t in texts]
    df: dict[str, int] = defaultdict(int)
    for tok_list in docs:
        for tok in set(tok_list):
            df[tok] += 1
    vocab = {tok: idx for idx, tok in enumerate(sorted(df))}
    n_docs = len(docs)

    vectors: list[list[float]] = []
    for tok_list in docs:
        vec = [0.0] * len(vocab)
        counts: dict[str, int] = defaultdict(int)
        for tok in tok_list:
            counts[tok] += 1
        for tok, cnt in counts.items():
            idf = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
            vec[vocab[tok]] = (1.0 + math.log(cnt)) * idf
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


def _kmeans_cosine(vectors: list[list[float]], k: int, iters: int = 25) -> list[int]:
    """Simple deterministic spherical k-means requiring no external ML libraries."""
    n = len(vectors)
    dim = len(vectors[0])
    # Deterministic farthest-first initialization
    centroids = [list(vectors[0])]
    chosen = {0}
    for _ in range(1, k):
        best_idx = 0
        best_sim = float("inf")
        for i, v in enumerate(vectors):
            if i in chosen:
                continue
            max_sim = max(sum(a * b for a, b in zip(v, c)) for c in centroids)
            if max_sim < best_sim:
                best_sim = max_sim
                best_idx = i
        chosen.add(best_idx)
        centroids.append(list(vectors[best_idx]))

    assignments = [0] * n
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            best_c = max(range(k), key=lambda c: sum(a * b for a, b in zip(v, centroids[c])))
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True
        if not changed:
            break
        for c in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == c]
            if not members:
                continue
            mean = [sum(m[d] for m in members) for d in range(dim)]
            norm = math.sqrt(sum(x * x for x in mean)) or 1.0
            centroids[c] = [x / norm for x in mean]
    return assignments


def build_clusters(
    provider: LLMProvider,
    definitions: dict[str, ClassDefinition],
    *,
    num_clusters: int | None = None,
    embed_model: str | None = None,
    max_workers: int = 8,
) -> dict[str, ClusterNode]:
    """Cluster class definitions semantically and synthesize super-class summaries."""
    labels = list(definitions.keys())
    k = num_clusters or max(2, min(15, round(math.sqrt(len(labels)))))
    k = min(k, len(labels))

    texts = [f"{definitions[lbl].semantic_name}: {definitions[lbl].definition}" for lbl in labels]
    vectors = _embed_definitions(texts, embed_model)
    assignments = _kmeans_cosine(vectors, k=k)

    grouped: dict[str, list[str]] = defaultdict(list)
    for lbl, cid in zip(labels, assignments):
        grouped[f"C{cid + 1}"].append(lbl)

    def _summarize(cid: str, members: list[str]) -> ClusterNode:
        blurbs = "\n".join(
            f"- {definitions[m].semantic_name}: {definitions[m].definition}" for m in members
        )
        prompt = (
            f"Summarize this cluster of {len(members)} related classes for coarse Stage-1 routing.\n\n"
            f"Member classes:\n{blurbs}\n\n"
            f"Write a self-sufficient summary covering ALL member intents without referencing "
            f"internal label IDs."
        )
        raw = provider.complete_json(
            prompt,
            ClusterSummarySchema.model_json_schema(),
            temperature=0.2,
        )
        parsed = ClusterSummarySchema.model_validate(raw)
        signals = "\n".join(f"  - {s}" for s in parsed.key_signals)
        summary_text = (
            f"Cluster: {cid}\n"
            f"Summary: {parsed.summary_definition}\n"
            f"Covers requests about:\n{signals}"
        )
        return ClusterNode(
            cluster_id=cid,
            summary_name=parsed.summary_name,
            summary_text=summary_text,
            members=members,
        )

    nodes: dict[str, ClusterNode] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(grouped))) as pool:
        futs = {pool.submit(_summarize, cid, m): cid for cid, m in grouped.items()}
        for fut in as_completed(futs):
            node = fut.result()
            nodes[node.cluster_id] = node

    return {cid: nodes[cid] for cid in sorted(nodes, key=lambda x: int(x[1:]))}
