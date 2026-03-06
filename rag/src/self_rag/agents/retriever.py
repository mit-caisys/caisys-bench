"""
Retriever agent for Self-RAG workflow.

Creates vector store and retrieval components using Milvus.
"""

import sys
from pathlib import Path

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import MilvusClient

from .utils import get_documents, get_embedding

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import EMBEDDING_CACHE_DIR, MILVUS_PATH


def create_vector_store(
    retriever_config,
    documents_config,
    milvus_path: str = str(MILVUS_PATH),
    embedding_cache_dir: Path = EMBEDDING_CACHE_DIR,
):
    embedding = get_embedding(retriever_config.embedding)
    client = MilvusClient(milvus_path)

    collection_name = documents_config.collection_name
    connection_args = {"uri": milvus_path}
    index_params = {
        "metric_type": "L2",
        "index_type": "FLAT",
        "params": {},
    }

    if not documents_config.override and client.has_collection(
        collection_name=collection_name
    ):
        client.close()
        milvus = Milvus(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            index_params=index_params,
        )
        if retriever_config.increase_latency:
            milvus.col.release()
        return milvus
    else:
        client.close()
        documents = get_documents(documents_config.urls)

        if retriever_config.split:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=retriever_config.chunk_size,
                chunk_overlap=retriever_config.chunk_overlap,
            )
            documents = text_splitter.split_documents(documents)

        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_embedding = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=LocalFileStore(embedding_cache_dir),
            key_encoder="sha512",  # pyright: ignore
            namespace=f"{retriever_config.embedding.provider}",
        )

        limit = 100
        milvus = Milvus.from_documents(
            documents=documents[:limit],
            embedding=cached_embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            index_params=index_params,
            drop_old=True,
        )

        for chunk in range(limit, len(documents), limit):
            milvus.add_documents(documents[chunk : chunk + limit])

        if retriever_config.increase_latency:
            milvus.col.release()
        return milvus
