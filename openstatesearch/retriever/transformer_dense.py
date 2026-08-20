from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import Document, SearchHit


QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        last_hidden_states.new_tensor(range(batch_size), dtype=sequence_lengths.dtype),
        sequence_lengths,
    ]


class TransformerDenseRetriever:
    """Frozen Qwen/LRAT dense index for production retrieval."""

    def __init__(
        self,
        documents: list[Document],
        model_path: str,
        index_path: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        dtype: str = "auto",
    ):
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("install openstatesearch[retriever] for LRAT dense search") from exc
        self.np = np
        self.torch = torch
        self.documents = {document.doc_id: document for document in documents}
        self.doc_ids = sorted(self.documents)
        self._doc_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model_dtype = None
        if dtype == "bfloat16":
            model_dtype = torch.bfloat16
        elif dtype == "float16":
            model_dtype = torch.float16
        elif dtype != "auto":
            raise ValueError(f"unsupported dtype: {dtype}")
        load_kwargs = {"torch_dtype": model_dtype} if model_dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs).to(self.device).eval()
        self._device_vectors = None
        if index_path:
            stored = np.load(index_path, allow_pickle=False)
            stored_ids = [str(value) for value in stored["doc_ids"]]
            if stored_ids != self.doc_ids:
                raise ValueError("dense index doc_ids do not match the frozen corpus")
            stored_vectors = stored["vectors"]
            if str(self.device).startswith("cuda"):
                self.vectors = stored_vectors
                self._device_vectors = torch.from_numpy(
                    stored_vectors.astype("float16", copy=False)
                ).to(self.device)
            else:
                self.vectors = stored_vectors.astype("float32")
        else:
            self.vectors = self.encode_documents()

    def _encode(self, texts: list[str], max_length: int) -> Any:
        torch = self.torch
        with torch.inference_mode():
            batch = self.tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch).last_hidden_state
            embeddings = last_token_pool(outputs, batch["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            return embeddings.float().cpu().numpy()

    def encode_documents(self, batch_size: int | None = None) -> Any:
        batch_size = batch_size or self.batch_size
        batches = []
        for start in range(0, len(self.doc_ids), batch_size):
            ids = self.doc_ids[start : start + batch_size]
            texts = [
                f"{self.documents[doc_id].title}\n{self.documents[doc_id].text}" for doc_id in ids
            ]
            batches.append(self._encode(texts, 512))
        return self.np.concatenate(batches, axis=0)

    def save_index(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("dense index path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.np.savez_compressed(
            destination,
            doc_ids=self.np.asarray(self.doc_ids),
            vectors=self.vectors.astype("float16"),
        )
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "index": str(destination),
            "sha256": digest,
            "document_count": len(self.doc_ids),
            "doc_ids_sha256": hashlib.sha256("\n".join(self.doc_ids).encode("utf-8")).hexdigest(),
            "dimensions": int(self.vectors.shape[1]),
            "pooling": "last_token",
            "query_instruction": QUERY_INSTRUCTION,
        }
        Path(f"{destination}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def search(self, query: str, k: int = 100) -> list[SearchHit]:
        query_vector = self._encode([format_query(query)], 128)[0]
        if self._device_vectors is not None:
            device_query = self.torch.from_numpy(query_vector).to(
                self.device, dtype=self._device_vectors.dtype
            )
            scores = self.torch.mv(self._device_vectors, device_query).float().cpu().numpy()
        else:
            scores = self.vectors @ query_vector
        if k <= 0:
            return []
        limit = min(k, len(scores))
        if limit == len(scores):
            order = self.np.argsort(-scores, kind="stable")
        else:
            candidates = self.np.argpartition(scores, -limit)[-limit:]
            order = candidates[self.np.lexsort((candidates, -scores[candidates]))]
        return [
            SearchHit(
                self.doc_ids[index],
                self.documents[self.doc_ids[index]].title,
                float(scores[index]),
                self.documents[self.doc_ids[index]].text[:512],
                self.documents[self.doc_ids[index]].source,
            )
            for index in order
        ]

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def search_batch(
        self, queries: list[str], k: int = 100, batch_size: int = 64
    ) -> list[list[SearchHit]]:
        """Batched query encoding/search with exact stable top-k ordering."""
        if k <= 0:
            return [[] for _ in queries]
        rankings: list[list[SearchHit]] = []
        for start in range(0, len(queries), batch_size):
            texts = [format_query(query) for query in queries[start : start + batch_size]]
            query_vectors = self._encode(texts, 128)
            if self._device_vectors is not None:
                device_queries = self.torch.from_numpy(query_vectors).to(
                    self.device, dtype=self._device_vectors.dtype
                )
                score_matrix = (device_queries @ self._device_vectors.T).float().cpu().numpy()
            else:
                score_matrix = query_vectors @ self.vectors.T
            limit = min(k, len(self.doc_ids))
            for scores in score_matrix:
                if limit == len(scores):
                    order = self.np.argsort(-scores, kind="stable")
                else:
                    candidates = self.np.argpartition(scores, -limit)[-limit:]
                    order = candidates[self.np.lexsort((candidates, -scores[candidates]))]
                rankings.append(
                    [
                        SearchHit(
                            self.doc_ids[index],
                            self.documents[self.doc_ids[index]].title,
                            float(scores[index]),
                            self.documents[self.doc_ids[index]].text[:512],
                            self.documents[self.doc_ids[index]].source,
                        )
                        for index in order
                    ]
                )
        return rankings
