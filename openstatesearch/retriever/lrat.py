from __future__ import annotations

from typing import Any


def clipped_relevance_weight(reweight_rate: float) -> float:
    return min(2.0, max(0.5, float(reweight_rate)))


def weighted_contrastive_loss(
    query_embeddings: Any,
    document_embeddings: Any,
    positive_indices: Any,
    weights: Any,
    temperature: float = 0.02,
) -> Any:
    """LRAT weighted in-batch contrastive loss with lazy PyTorch import.

    Embeddings must be shaped [batch, hidden], positive_indices [batch], and weights [batch].
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("install openstatesearch[retriever] to compute LRAT loss") from exc

    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 tensors")
    queries = functional.normalize(query_embeddings, dim=-1)
    documents = functional.normalize(document_embeddings, dim=-1)
    logits = queries @ documents.transpose(0, 1) / temperature
    per_example = functional.cross_entropy(logits, positive_indices, reduction="none")
    clipped = torch.clamp(weights.to(per_example.dtype), 0.5, 2.0)
    return (per_example * clipped).mean()
