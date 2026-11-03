"""Vector store adapters: pgvector, Chroma, Qdrant, and Milvus."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import VectorStoreType, settings
from app.models.knowledge import Embedding, KnowledgeBase
from app.providers.ai.exceptions import ProviderError


@dataclass(slots=True)
class VectorRecord:
    chunk_id: str
    vector: list[float]
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorMatch:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    async def upsert(self, records: list[VectorRecord]) -> None: ...

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]: ...

    async def delete(self, chunk_ids: list[str]) -> None: ...

    async def count(self) -> int: ...

    async def aclose(self) -> None: ...


class PgVectorStore:
    """Stores vectors in the database; uses pgvector on PostgreSQL."""

    def __init__(
        self,
        session: AsyncSession,
        knowledge_base_id: str,
        *,
        provider: str,
        model: str,
        dimensions: int,
    ) -> None:
        self.session = session
        self.knowledge_base_id = knowledge_base_id
        self.provider = provider
        self.model = model
        self.dimensions = dimensions

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        await self.delete([record.chunk_id for record in records])
        for record in records:
            self.session.add(
                Embedding(
                    chunk_id=record.chunk_id,
                    knowledge_base_id=self.knowledge_base_id,
                    provider=self.provider,
                    model=self.model,
                    dimensions=len(record.vector),
                    vector_store="pgvector",
                    external_id=record.chunk_id,
                    vector=record.vector,
                    norm=math.sqrt(sum(value * value for value in record.vector)),
                    metadata_=record.metadata or None,
                )
            )
        await self.session.flush()

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if settings.is_postgres:
            try:
                distance = Embedding.vector.cosine_distance(vector)
                stmt = (
                    select(Embedding.chunk_id, (1 - distance).label("score"))
                    .where(Embedding.knowledge_base_id == self.knowledge_base_id)
                    .order_by(distance)
                    .limit(max(limit * 4, limit))
                )
                rows = (await self.session.execute(stmt)).all()
                matches = [
                    VectorMatch(chunk_id=row[0], score=float(row[1] or 0.0))
                    for row in rows
                ]
                return self._filter(matches, metadata_filter)[:limit]
            except Exception:
                await self.session.rollback()
        stmt = select(Embedding).where(
            Embedding.knowledge_base_id == self.knowledge_base_id
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        matches: list[VectorMatch] = []
        for embedding in rows:
            candidate = embedding.vector
            if candidate is None:
                continue
            if isinstance(candidate, str):
                import json

                try:
                    candidate = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            score = _cosine(vector, [float(value) for value in candidate])
            matches.append(VectorMatch(chunk_id=embedding.chunk_id, score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return self._filter(matches, metadata_filter)[:limit]

    def _filter(
        self,
        matches: list[VectorMatch],
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorMatch]:
        # Row-level metadata filtering is applied in RAGService where both the
        # document and chunk metadata are available.  Accepting the argument
        # here keeps the adapter contract uniform across backends.
        return matches

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        await self.session.execute(
            delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids))
        )
        await self.session.flush()

    async def count(self) -> int:
        total = await self.session.scalar(
            select(func.count())
            .select_from(Embedding)
            .where(Embedding.knowledge_base_id == self.knowledge_base_id)
        )
        return int(total or 0)

    async def aclose(self) -> None:
        return None


class ChromaVectorStore:
    def __init__(
        self,
        knowledge_base_id: str,
        *,
        dimensions: int,
        host: str,
        port: int,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "chromadb is required for VECTOR_STORE_TYPE=chroma"
            ) from exc
        self.dimensions = dimensions
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=f"kb_{knowledge_base_id}",
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[record.chunk_id for record in records],
            embeddings=[record.vector for record in records],
            documents=[record.content for record in records],
            metadatas=[record.metadata or {} for record in records],
        )

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[vector],
            n_results=limit,
            where=metadata_filter or None,
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            VectorMatch(chunk_id=chunk_id, score=max(0.0, 1.0 - float(distance)))
            for chunk_id, distance in zip(ids, distances, strict=False)
        ]

    async def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            await asyncio.to_thread(self._collection.delete, ids=chunk_ids)

    async def count(self) -> int:
        return int(await asyncio.to_thread(self._collection.count))

    async def aclose(self) -> None:
        return None


class QdrantVectorStore:
    def __init__(
        self,
        knowledge_base_id: str,
        *,
        dimensions: int,
        host: str,
        port: int,
        api_key: str | None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "qdrant-client is required for VECTOR_STORE_TYPE=qdrant"
            ) from exc
        self.collection = f"kb_{knowledge_base_id}"
        self.dimensions = dimensions
        self._client = QdrantClient(host=host, port=port, api_key=api_key)
        self._create(dimensions=dimensions, distance=Distance, params=VectorParams)

    def _create(self, *, dimensions: int, distance: Any, params: Any) -> None:
        try:
            collections = {
                item.name for item in self._client.get_collections().collections
            }
            if self.collection not in collections:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=params(size=dimensions, distance=distance.COSINE),
                )
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(f"Could not initialise Qdrant: {exc}") from exc

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_stable_uuid(record.chunk_id),
                vector=record.vector,
                payload={"chunk_id": record.chunk_id, **record.metadata},
            )
            for record in records
        ]
        await asyncio.to_thread(
            self._client.upsert, collection_name=self.collection, points=points
        )

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if metadata_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )
        response = await asyncio.to_thread(
            self._client.search,
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )
        return [
            VectorMatch(
                chunk_id=str((point.payload or {}).get("chunk_id") or point.id),
                score=float(point.score),
            )
            for point in response
        ]

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        from qdrant_client.models import PointIdsList

        await asyncio.to_thread(
            self._client.delete,
            collection_name=self.collection,
            points_selector=PointIdsList(points=[_stable_uuid(cid) for cid in chunk_ids]),
        )

    async def count(self) -> int:
        result = await asyncio.to_thread(
            self._client.count, collection_name=self.collection
        )
        return int(result.count)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)


class MilvusVectorStore:
    def __init__(
        self,
        knowledge_base_id: str,
        *,
        dimensions: int,
        host: str,
        port: int,
    ) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "pymilvus is required for VECTOR_STORE_TYPE=milvus. "
                "Install it with: pip install pymilvus"
            ) from exc
        self.collection = f"kb_{knowledge_base_id}"
        self.dimensions = dimensions
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if not self._client.has_collection(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                dimension=dimensions,
                metric_type="COSINE",
                auto_id=False,
                primary_field_name="id",
                vector_field_name="vector",
                id_type="string",
                max_length=64,
            )

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        rows = [
            {
                "id": record.chunk_id,
                "vector": record.vector,
                **{
                    key: value
                    for key, value in record.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
            for record in records
        ]
        await asyncio.to_thread(
            self._client.upsert, collection_name=self.collection, data=rows
        )

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        expression = None
        if metadata_filter:
            clauses = [
                f'{key} == "{value}"' if isinstance(value, str) else f"{key} == {value}"
                for key, value in metadata_filter.items()
            ]
            expression = " and ".join(clauses)
        results = await asyncio.to_thread(
            self._client.search,
            collection_name=self.collection,
            data=[vector],
            limit=limit,
            filter=expression or "",
            output_fields=["id"],
        )
        matches: list[VectorMatch] = []
        for row in (results[0] if results else []):
            entity = row.get("entity") or {}
            matches.append(
                VectorMatch(
                    chunk_id=str(entity.get("id") or row.get("id")),
                    score=float(row.get("distance") or 0.0),
                )
            )
        return matches

    async def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self.collection,
                ids=chunk_ids,
            )

    async def count(self) -> int:
        stats = await asyncio.to_thread(
            self._client.get_collection_stats, collection_name=self.collection
        )
        return int(stats.get("row_count") or 0)

    async def aclose(self) -> None:
        return None


def get_vector_store(
    session: AsyncSession, knowledge_base: KnowledgeBase
) -> VectorStore:
    dimensions = knowledge_base.embedding_dimensions or settings.embedding_dimensions
    store_type = settings.vector_store_type
    if store_type == VectorStoreType.CHROMA:
        return ChromaVectorStore(
            knowledge_base.id,
            dimensions=dimensions,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    if store_type == VectorStoreType.QDRANT:
        return QdrantVectorStore(
            knowledge_base.id,
            dimensions=dimensions,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
    if store_type == VectorStoreType.MILVUS:
        return MilvusVectorStore(
            knowledge_base.id,
            dimensions=dimensions,
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
    return PgVectorStore(
        session,
        knowledge_base.id,
        provider=knowledge_base.embedding_provider,
        model=knowledge_base.embedding_model,
        dimensions=dimensions,
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _stable_uuid(value: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


__all__ = [
    "ChromaVectorStore",
    "MilvusVectorStore",
    "PgVectorStore",
    "QdrantVectorStore",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "get_vector_store",
]
