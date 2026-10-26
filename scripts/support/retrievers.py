"""BM25, dense retrieval with cached embeddings, and random sampling."""

from collections import Counter
from pathlib import Path
import re

import numpy as np

from .common import digest
from . import providers


def tokenize(text):
    """Lowercase text into ASCII alphanumeric tokens for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25(pool, query, full=False):
    """Rank headlines or full articles with BM25 (k1=1.5, b=0.75)."""
    texts = pool.headline + ("\n" + pool.body if full else "")
    documents = [tokenize(text) for text in texts]
    counts = [Counter(tokens) for tokens in documents]
    lengths = np.array([len(tokens) for tokens in documents])
    document_frequency = Counter(token for tokens in documents for token in set(tokens))
    scores = np.zeros(len(pool))
    for token in tokenize(query):
        frequency = np.array([document[token] for document in counts])
        df = document_frequency[token]
        idf = np.log(1 + (len(pool) - df + .5) / (df + .5))
        scores += idf * frequency * 2.5 / (
            frequency + 1.5 * (.25 + .75 * lengths / lengths.mean())
        )
    # Stable sorting breaks tied scores in corpus order.
    return pool.iloc[np.argsort(-scores, kind="stable")[:10]].index.tolist()


class Embeddings:
    """Load saved vectors, requesting and saving any missing embeddings."""

    def __init__(self, path="cache/embeddings"):
        """Load embedding shards if present; an absent folder starts an empty cache."""
        self.path = Path(path)
        self.vectors = {}
        for file in sorted(self.path.glob("*.npz")):
            with np.load(file) as stored:
                self.vectors.update(zip(stored["keys"], stored["vectors"]))
        self.hits = self.calls = 0

    def get(self, text):
        """Return a unit-length embedding, requesting and saving it on a cache miss."""
        key = digest(dict(
            stage="embedding", model="google/gemini-embedding-2", text=text, dimensions=3072,
        ))
        if key in self.vectors:
            self.hits += 1
        else:
            vector = np.asarray(providers.embed(text), dtype=float)
            if vector.shape != (3072,) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
                raise ValueError("The provider returned an invalid embedding.")
            self.path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path / f"{key}.npz", keys=[key], vectors=[vector])
            self.vectors[key] = vector
            self.calls += 1
        vector = self.vectors[key]
        return vector / np.linalg.norm(vector)


def dense(pool, query, cache, full=False):
    """Return the ten closest article IDs using headlines or headlines plus body chunks."""
    # Prefixes and the 1,000-word body limit are part of the saved embedding requests.
    texts = [
        f"title: none | text: {row.headline}"
        + ("\n\n" + " ".join(row.body.split()[:1000]) if full else "")
        for row in pool.itertuples()
    ]
    # Cached vectors are normalized, so their dot product is cosine similarity.
    matrix = np.stack([cache.get(text) for text in texts])
    query_vector = cache.get(f"task: search result | query: {query}")
    return pool.iloc[np.argsort(-(matrix @ query_vector), kind="stable")[:10]].index.tolist()


def uniform_sets(rng, population, count, k):
    """Sample ordered sets, with no repeated article within a set.

    Sampling one column at a time preserves the paper's random-number sequence.
    """
    picks = np.empty((count, k), dtype=int)
    for column in range(k):
        picks[:, column] = rng.integers(population, size=count)
        if column:
            duplicate = (picks[:, :column] == picks[:, column, None]).any(axis=1)
            while duplicate.any():
                picks[duplicate, column] = rng.integers(population, size=duplicate.sum())
                duplicate = (picks[:, :column] == picks[:, column, None]).any(axis=1)
    return picks
