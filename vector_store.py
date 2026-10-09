import os
import sys
import json
import hashlib
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types


class LocalVectorStore:
    """Manages chunk embeddings, disk caching, and hybrid cosine similarity search."""

    def __init__(
        self,
        client: Optional[genai.Client],
        model_name: str = "gemini-embedding-001",
        cache_dir: Optional[Path] = None
    ):
        self.client = client
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".rag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, D)

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Computes SHA256 of file for cache invalidation."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_or_load_index(
        self,
        pdf_path: Path,
        chunks: List[Dict[str, Any]],
        force_rebuild: bool = False
    ) -> int:
        """Loads index from cache if valid, otherwise computes embeddings and saves cache.

        Returns:
            The number of indexed chunks.
        """
        self.chunks = chunks
        file_hash = self._compute_file_hash(pdf_path)
        cache_meta_file = self.cache_dir / f"{pdf_path.stem}_{file_hash[:12]}.json"
        cache_vec_file = self.cache_dir / f"{pdf_path.stem}_{file_hash[:12]}.npy"

        # Check existing valid cache
        if not force_rebuild and cache_meta_file.exists() and cache_vec_file.exists():
            try:
                with open(cache_meta_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("file_hash") == file_hash and len(cached_data.get("chunks", [])) == len(chunks):
                    self.chunks = cached_data["chunks"]
                    self.embeddings = np.load(cache_vec_file)
                    print(f"⚡ Loaded {len(self.chunks)} vector embeddings from cache.")
                    return len(self.chunks)
            except Exception as e:
                print(f"⚠️ Cache read error ({e}), recomputing embeddings...")

        if not self.client:
            print("⚠️ Gemini client not initialized. Falling back to keyword-only indexing.")
            return len(self.chunks)

        print(f"🔄 Computing vector embeddings for {len(chunks)} chunks using '{self.model_name}'...")
        all_vectors: List[List[float]] = []
        batch_size = 15

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c["text"] for c in batch]
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts
                )
                for emb in response.embeddings:
                    all_vectors.append(emb.values)
                print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r", flush=True)
            except Exception as e:
                print(f"\n❌ Error embedding batch {i}-{i+batch_size}: {e}")
                raise e

        print(f"\n✅ Finished generating {len(all_vectors)} embeddings.")
        self.embeddings = np.array(all_vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity (dot product of normalized vectors = cosine sim)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = self.embeddings / norms

        # Save to cache
        try:
            with open(cache_meta_file, "w", encoding="utf-8") as f:
                json.dump({"file_hash": file_hash, "chunks": self.chunks}, f, ensure_ascii=False)
            np.save(cache_vec_file, self.embeddings)
            print(f"💾 Cached embeddings to {cache_vec_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to write cache: {e}")

        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic similarity search with keyword bonus scoring."""
        if not self.chunks:
            return []

        scores = np.zeros(len(self.chunks), dtype=np.float32)

        # 1. Semantic Embedding Search
        if self.client is not None and self.embeddings is not None:
            try:
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=query
                )
                q_vec = np.array(res.embeddings[0].values, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm
                    # Dot product with normalized chunk embeddings = cosine similarity
                    scores = np.dot(self.embeddings, q_vec)
            except Exception as e:
                print(f"⚠️ Embedding query failed ({e}), using lexical ranking...")

        # 2. Lexical / Keyword Matching Boost
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk["text"].lower()
            term_matches = sum(1 for term in query_terms if term in chunk_lower)
            if term_matches > 0:
                # Add lexical bonus
                scores[idx] += 0.05 * (term_matches / max(len(query_terms), 1))

        # Rank top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "id": self.chunks[idx]["id"],
                "page_number": self.chunks[idx]["page_number"],
                "score": float(scores[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results
