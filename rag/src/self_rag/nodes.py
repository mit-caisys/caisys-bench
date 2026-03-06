"""
Self-RAG workflow node implementations.

Defines the operations for retrieve, grade, generate, and rewrite nodes.
"""

import logging
import time
from typing import Dict

from langchain_core.callbacks import get_usage_metadata_callback
from self_rag.agents.utils import get_num_tokens

from .timeline import TimelineTracer
from .utils import opt_print, opt_print_factory


def search_vector_store(
    vectorstore,
    embedding: list[float],
    search_type: str,
    search_kwargs: Dict,
    increase_retrieval_latency: bool,
):
    if increase_retrieval_latency:
        vectorstore.col.load()

    match search_type:
        case "similarity":
            result = vectorstore.similarity_search_by_vector(
                embedding=embedding, **search_kwargs
            )
        case "similarity_score_threshold":
            result = vectorstore.similarity_search_with_score_by_vector(
                embedding=embedding, **search_kwargs
            )
        case "mmr":
            result = vectorstore.max_marginal_relevance_search_by_vector(
                embedding=embedding, **search_kwargs
            )
        case _:
            raise KeyError(f"Missing search_type {search_type}")

    if increase_retrieval_latency:
        vectorstore.col.release()
    return result


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    with TimelineTracer(state["request_context"].timeline, "retrieve", "node"):
        start_time = state["start_time"] if state["start_time"] > 0.0 else time.time()
        logging.info(f"{state['request_id']} start retrieve")
        opt_print(state["verbose"], "---RETRIEVE---\n")
        question_embedding = state["embedding"].embed_query(state["question"])
        documents = search_vector_store(
            state["vector_store"],
            question_embedding,
            state["search_type"],
            state["search_kwargs"],
            state["increase_retrieval_latency"],
        )

        logging.info(f"{state['request_id']} end retrieve")
        return {
            "documents": documents,
            "nodes_path": state["nodes_path"] + ["retrieve"],
            "start_time": start_time,
            "end_time": time.time(),
        }


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    with TimelineTracer(state["request_context"].timeline, "generate", "node"):
        logging.info(f"{state['request_id']} start generate")
        opt_print(state["verbose"], "---GENERATE---\n")

        with get_usage_metadata_callback() as cb:
            generation = state["generator"].invoke(
                {"context": state["documents"], "question": state["question"]},
            )

            logging.info(f"{state['request_id']} end generate")
            return {
                "generation": generation,
                "generate_iteration_count": state["generate_iteration_count"] + 1,
                "input_tokens": state["input_tokens"]
                + [get_num_tokens("input_tokens", cb.usage_metadata)],
                "output_tokens": state["output_tokens"]
                + [get_num_tokens("output_tokens", cb.usage_metadata)],
                "nodes_path": state["nodes_path"] + ["generate"],
                "start_time": state["start_time"],
                "end_time": time.time(),
            }


def grade(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    with TimelineTracer(state["request_context"].timeline, "grade", "node"):
        logging.info(f"{state['request_id']} start grade")
        opt_print_local = opt_print_factory(state["verbose"])
        opt_print_local("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        # Rate each document
        filtered_docs = []
        decisions = []

        with get_usage_metadata_callback() as cb:
            for d in state["documents"]:
                grade = (
                    "yes"
                    if state["retrieval_grader"]["skip"]
                    else state["retrieval_grader"]["runnable"]
                    .invoke({"question": state["question"], "document": d.page_content})
                    .binary_score
                )
                if grade == "yes":
                    opt_print_local("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                    decisions.append("Relevant")
                else:
                    opt_print_local("---GRADE: DOCUMENT NOT RELEVANT---")
                    decisions.append("Not relevant")

            opt_print_local("")

            logging.info(f"{state['request_id']} end grade")
            return {
                "documents": filtered_docs,
                "input_tokens": state["input_tokens"]
                + [get_num_tokens("input_tokens", cb.usage_metadata)],
                "output_tokens": state["output_tokens"]
                + [get_num_tokens("output_tokens", cb.usage_metadata)],
                "nodes_path": state["nodes_path"] + ["grade"],
                "start_time": state["start_time"],
                "end_time": time.time(),
            }


def rewrite_question(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    with TimelineTracer(state["request_context"].timeline, "rewrite_question", "node"):
        logging.info(f"{state['request_id']} start rewrite_question")
        opt_print(state["verbose"], "---REWRITE QUESTION---\n")

        with get_usage_metadata_callback() as cb:
            better_question = state["question_rewriter"].invoke(
                {"question": state["question"]}
            )

            logging.info(f"{state['request_id']} end rewrite_question")
            return {
                "question": better_question,
                "rewrite_iteration_count": state["rewrite_iteration_count"] + 1,
                "generate_iteration_count": 0,
                "input_tokens": state["input_tokens"]
                + [get_num_tokens("input_tokens", cb.usage_metadata)],
                "output_tokens": state["output_tokens"]
                + [get_num_tokens("output_tokens", cb.usage_metadata)],
                "nodes_path": state["nodes_path"] + ["rewrite_question"],
                "start_time": state["start_time"],
                "end_time": time.time(),
            }


def cannot_answer(state):
    """
    Change the generated answer to indicate that it cannot answer the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Change generation to indicate that it cannot generate answer
    """

    with TimelineTracer(state["request_context"].timeline, "cannot_answer", "node"):
        logging.info(f"{state['request_id']} start cannot_answer")
        opt_print(state["verbose"], "---CANNOT ANSWER---\n")
        logging.info(f"{state['request_id']} end cannot_answer")
        return {
            "generation": "Cannot answer the question based on the retrieved documents",
            "nodes_path": state["nodes_path"] + ["cannot_answer"],
            "start_time": state["start_time"],
            "end_time": time.time(),
        }
