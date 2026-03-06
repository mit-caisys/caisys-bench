"""
Vector database creation script for RAG.

Creates and populates Milvus vector store with Wikipedia document embeddings.
"""

import argparse
import ast
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import pandas as pd
from box import Box
from common.paths import DATASET_DIR, EMBEDDING_CACHE_DIR, FRAMES_PATH, MILVUS_PATH
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from self_rag.agents.utils import get_documents, get_embedding

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script for creating large vectorstore"
    )

    parser.add_argument(
        "-i",
        "--initialize",
        action="store_true",
        help="Initialize the vectorstore with frames data",
    )

    args = parser.parse_args()
    return args


def get_milvus(collection_name, retriever_config, milvus_path, embedding_cache_dir):
    embedding = get_embedding(retriever_config.embedding)
    client = MilvusClient(milvus_path)

    if client.has_collection(collection_name):
        client.close()
        milvus = Milvus(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args={"uri": milvus_path},
        )
    else:
        client.close()
        cached_embedding = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=LocalFileStore(embedding_cache_dir),
            key_encoder="sha512",  # pyright: ignore
            namespace=f"{retriever_config.embedding.provider}",
        )
        index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}

        milvus = Milvus.from_documents(
            documents=[],
            embedding=cached_embedding,
            collection_name=collection_name,
            connection_args={"uri": milvus_path},
            index_params=index_params,
            drop_old=True,
        )

    return milvus


def get_urls_chunk(milvus_path, checkpoint_file, chunk_size=20, max_lines=10000):
    start_line = 0
    shard = "data-00000-of-00010"
    shard_no = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.readline().strip())
            shard_no = int(f.readline().strip())
            shard = f"data-{shard_no:05d}-of-00010"

    blocks_in_chunk = 0
    unique_urls = set()
    current_block_urls = set()

    with open(Path(milvus_path) / shard, "r", encoding="utf-8") as f:
        list(islice(f, start_line))
        for relative_line in range(max_lines):
            line = f.readline()
            if not line:
                unique_urls.update(current_block_urls)
                yield list(unique_urls)

                with open(checkpoint_file, "w") as cp:
                    cp.write(f"0\n{shard_no + 1}\n")
                break

            clean_line = line.strip()

            if clean_line.startswith("MENTION"):
                parts = clean_line.split()
                if parts and parts[-1].startswith("http"):
                    current_block_urls.add(parts[-1])

            if clean_line == "":
                if current_block_urls:
                    unique_urls.update(current_block_urls)
                    current_block_urls = set()
                    blocks_in_chunk += 1

                if blocks_in_chunk >= chunk_size:
                    yield list(unique_urls)
                    unique_urls = set()
                    blocks_in_chunk = 0

                    with open(checkpoint_file, "w") as cp:
                        cp.write(f"{relative_line + start_line + 1}\n{shard_no}\n")


def process_chunk_sync(urls_chunk, retriever_config):
    documents = get_documents(urls_chunk)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=retriever_config.chunk_size,
        chunk_overlap=retriever_config.chunk_overlap,
    )
    return text_splitter.split_documents(documents)


async def run_single_job(urls_chunk, retriever_config, milvus, pool):
    loop = asyncio.get_running_loop()

    try:
        docs = await loop.run_in_executor(
            pool,
            partial(process_chunk_sync, urls_chunk, retriever_config),
        )

        if docs:
            try:
                await milvus.aadd_documents(docs)
            except Exception as e:
                logger.error(f"Failed to insert documents into Milvus: {e}")
        else:
            logger.warning("No documents were generated from this chunk.")

    except Exception as e:
        logger.error(f"Error during processing or executor task: {e}")


async def acreate_database(
    collection_name,
    retriever_config,
    milvus_path,
    wikilinks_path,
    embedding_cache_dir,
    checkpoint_path,
    batch_size=100,
):
    milvus = get_milvus(
        collection_name, retriever_config, milvus_path, embedding_cache_dir
    )
    tasks = []
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for urls_chunk in get_urls_chunk(wikilinks_path, checkpoint_path):
            tasks.append(run_single_job(urls_chunk, retriever_config, milvus, pool))

            if len(tasks) == batch_size:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)


def get_frames_urls_chunk(frames_path, chunk_size=30):
    df = pd.read_csv(frames_path, sep="\t")

    links = []
    for index in range(df.shape[0]):
        extracted_links = ast.literal_eval(df.iloc[index]["wiki_links"])
        extracted_links = [
            (link if link.startswith("https") else f"https://{link}")
            for link in extracted_links
        ]
        links.extend(extracted_links)
        if len(links) >= chunk_size:
            yield links
            links = []

    yield links


async def aadd_frames_data(
    collection_name,
    retriever_config,
    milvus_path,
    embedding_cache_dir,
    frames_path,
    batch_size=100,
):
    milvus = get_milvus(
        collection_name, retriever_config, milvus_path, embedding_cache_dir
    )
    tasks = []
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for urls_chunk in get_frames_urls_chunk(frames_path):
            tasks.append(run_single_job(urls_chunk, retriever_config, milvus, pool))

            if len(tasks) == batch_size:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    model = "google/embeddinggemma-300m"
    provider = "vllm"
    chunk_size = 1000
    chunk_overlap = 100

    collection_name = "wikilinks"
    retriever_config = Box(
        {
            "embedding": {"provider": provider, "model": model},
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    )
    milvus_path = str(MILVUS_PATH)
    embedding_cache_dir = str(EMBEDDING_CACHE_DIR)
    checkpoint_path = str(DATASET_DIR / "checkpoint.txt")
    wikilinks_path = str(DATASET_DIR / collection_name)
    frames_path = str(FRAMES_PATH)

    args = parse_arguments()
    if args.initialize:
        asyncio.run(
            aadd_frames_data(
                collection_name,
                retriever_config,
                milvus_path,
                embedding_cache_dir,
                frames_path,
            )
        )
    else:
        asyncio.run(
            acreate_database(
                collection_name,
                retriever_config,
                milvus_path,
                wikilinks_path,
                embedding_cache_dir,
                checkpoint_path,
            )
        )
