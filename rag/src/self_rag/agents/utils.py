from typing import Dict, Literal

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DEFAULT_TEMPERATURE = 0.2


def get_llm(llm_config: Dict):
    match llm_config["provider"]:
        case "ollama":
            return ChatOllama(
                model=llm_config["model"],
                temperature=llm_config.get("temperature", DEFAULT_TEMPERATURE),
                num_ctx=llm_config.get("num_ctx", 4096),
            )
        case "openai":
            return ChatOpenAI(
                model=llm_config["model"],
                temperature=llm_config.get("temperature", DEFAULT_TEMPERATURE),
            )
        case "huggingface":
            return ChatOpenAI(
                model=llm_config["model"],
                openai_api_key="EMPTY",  # pyright: ignore
                openai_api_base="http://localhost:8000/v1",  # pyright: ignore
                temperature=llm_config.get("temperature", DEFAULT_TEMPERATURE),
                stop=llm_config.get("stop", []),  # pyright: ignore
            )
        case _:
            raise KeyError(f"Cannot find LLM provider {llm_config['provider']}")


def get_embedding(embedding_config: Dict):
    match embedding_config["provider"]:
        case "ollama":
            return OllamaEmbeddings(model=embedding_config["model"])
        case "openai":
            return OpenAIEmbeddings(model=embedding_config["model"])
        case _:
            raise KeyError(
                f"Cannot find embedding provider {embedding_config['provider']}"
            )


def get_documents(urls: list[str]) -> list[Document]:
    documents = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in documents for item in sublist]


def get_num_tokens(key: Literal["input_tokens", "output_tokens"], usage_metadata):
    usage_metadata_list = list(usage_metadata.values())
    if len(usage_metadata_list) == 0:
        return 0
    return list(usage_metadata.values())[0].get(key, 0)
