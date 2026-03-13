import datetime
import sys
from ast import Dict
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .agents import *
from .edges import docs_relevant, hallucinations_and_answers_question
from .nodes import cannot_answer, generate, grade, retrieve, rewrite_question
from .timeline import RequestContext

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.header import (
    ACTUAL_ANSWER_HEADER,
    END_TIME_HEADER,
    EXPECTED_ANSWER_HEADER,
    NODES_PATH_HEADER,
    QUESTION_HEADER,
    REQUEST_ID_HEADER,
    START_TIME_HEADER,
    TOTAL_INPUT_TOKENS_HEADER,
    TOTAL_OUTPUT_TOKENS_HEADER,
)
from common.path import TIMELINE_DIR


class Grader(TypedDict):
    runnable: Runnable
    skip: bool


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents

        retriever: vector store retriever
        retrieval_grader: grader that checks if retrieved documents are relevant
        generator: generator that generates answers
        hallucination_grader: grader that checks for hallucination
        answer_grader: grader that checks if the generation answers the question
        question_rewriter: rewriter that rewrites the question for retrieval

        max_rewrite_iterations: max number allowed before termination
        max_hallucinate_iterations: max number allowed before rewriting question

        verbose: option to print each step
    """

    question: str
    generation: str
    documents: List[str]

    # retriever: VectorStore
    embedding: Embeddings
    search_type: str
    search_kwargs: Dict
    vector_store: VectorStore
    retrieval_grader: Grader
    generator: Runnable
    hallucination_grader: Grader
    answer_grader: Grader
    question_rewriter: Runnable

    # iterations
    max_rewrite_iterations: int
    rewrite_iteration_count: int
    max_generate_iterations: int
    generate_iteration_count: int

    # metrics
    input_tokens: list[int]
    output_tokens: list[int]

    request_context: RequestContext
    request_id: str
    nodes_path: list[str]

    start_time: float
    end_time: float

    answer: Optional[str]
    verbose: bool


def create_self_rag():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("cannot_answer", cannot_answer)

    # Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        docs_relevant,
        {
            "rewrite_question": "rewrite_question",
            "generate": "generate",
            "reach_max_rewrite_iter": "cannot_answer",
        },
    )
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        hallucinations_and_answers_question,
        {
            "hallucinate": "generate",
            "not useful": "rewrite_question",
            "reach_max_generate_iter": "rewrite_question",
            "reach_max_rewrite_iter": "cannot_answer",
            "useful": END,
        },
    )
    workflow.add_edge("cannot_answer", END)

    # Compile
    self_rag = workflow.compile()
    return self_rag


def create_input(config):
    return {
        "question": config["question"],
        "embedding": get_embedding(config["retriever"]["embedding"]),
        "search_type": config["retriever"]["search_type"],
        "search_kwargs": config["retriever"]["search_kwargs"],
        "vector_store": create_vector_store(config["retriever"], config["documents"]),
        "retrieval_grader": {
            "runnable": create_retrieval_grader(config["retrieval_grader"]),
            "skip": config["retrieval_grader"]["skip"],
        },
        "generator": create_generator(config["generator"]),
        "hallucination_grader": {
            "runnable": create_hallucination_grader(config["hallucination_grader"]),
            "skip": config["hallucination_grader"]["skip"],
        },
        "answer_grader": {
            "runnable": create_answer_grader(config["answer_grader"]),
            "skip": config["answer_grader"]["skip"],
        },
        "question_rewriter": create_question_rewriter(config["question_rewriter"]),
        "max_rewrite_iterations": config["iteration"]["max_rewrite_iterations"],
        "rewrite_iteration_count": 0,
        "max_generate_iterations": config["iteration"]["max_generate_iterations"],
        "generate_iteration_count": 0,
        "input_tokens": [],
        "output_tokens": [],
        "request_context": RequestContext(),
        "request_id": config["request_id"],
        "nodes_path": [],
        "answer": config.get("answer", None),
        "verbose": config["verbose"],
        "start_time": 0.0,
        "end_time": 0.0,
    }


def run_self_rag(self_rag, config, timeline=False):
    input = create_input(config)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"---Question---\n{config['question']}\n")

    final_value = {}
    for output in self_rag.stream(input):
        for value in output.values():
            final_value = value

    print(f"---Answer---\n{final_value['generation']}\n")

    if config["answer"]:
        print(f"---Expected---\n{config['answer']}\n")

    result = {
        REQUEST_ID_HEADER: config["request_id"],
        START_TIME_HEADER: final_value["start_time"],
        END_TIME_HEADER: final_value["end_time"],
        NODES_PATH_HEADER: ",".join(final_value["nodes_path"]),
        TOTAL_INPUT_TOKENS_HEADER: sum(final_value["input_tokens"]),
        TOTAL_OUTPUT_TOKENS_HEADER: sum(final_value["output_tokens"]),
        QUESTION_HEADER: config["question"],
        ACTUAL_ANSWER_HEADER: final_value["generation"],
        EXPECTED_ANSWER_HEADER: config["answer"],
    }

    if timeline:
        with open(
            TIMELINE_DIR / f"{config['workflow']}_{timestamp}.csv",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(input["request_context"].get_timeline_csv())

    return result
