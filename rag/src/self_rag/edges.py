"""
Self-RAG workflow edge routing functions.

Defines conditional logic for routing between nodes based on retrieval results.
"""

from langchain_core.callbacks import get_usage_metadata_callback
from self_rag.agents.utils import get_num_tokens

from .timeline import TimelineTracer
from .utils import opt_print_factory


def docs_relevant(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    with TimelineTracer(state["request_context"].timeline, "docs_relevant", "edge"):
        opt_print = opt_print_factory(state["verbose"])
        opt_print("---ASSESS GRADED DOCUMENTS---")

        filtered_documents = state["documents"]

        if filtered_documents:
            opt_print("---DECISION: GENERATE ANSWER---\n")
            return "generate"

        if state["rewrite_iteration_count"] < state["max_rewrite_iterations"]:
            opt_print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUESTION---\n"
            )
            return "rewrite_question"

        opt_print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION AND MAX REWRITE ITERATIONS REACHES, CANNOT ANSWER---\n"
        )
        return "reach_max_rewrite_iter"


def hallucinations_and_answers_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    with TimelineTracer(
        state["request_context"].timeline, "hallucinations_and_answers_question", "edge"
    ):
        opt_print = opt_print_factory(state["verbose"])
        opt_print("---CHECK HALLUCINATIONS---")

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        with get_usage_metadata_callback() as cb:
            try:
                hallucination_grade = (
                    "yes"
                    if state["hallucination_grader"]["skip"]
                    else state["hallucination_grader"]["runnable"]
                    .invoke(
                        {
                            "documents": documents,
                            "question": question,
                            "generation": generation,
                        }
                    )
                    .binary_score
                )

                if hallucination_grade == "yes":
                    opt_print(
                        "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---\n",
                    )
                    return answers_question_helper(state)

                if state["generate_iteration_count"] < state["max_generate_iterations"]:
                    opt_print(
                        "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---\n"
                    )
                    return "hallucinate"

                if state["rewrite_iteration_count"] < state["max_rewrite_iterations"]:
                    opt_print(
                        "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS AND MAX GENERATE ITERATIONS REACH, REWRITE QUESTION---\n"
                    )
                    return "reach_max_generate_iter"

                opt_print(
                    "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS AND MAX REWRITE ITERATIONS REACH, CANNOT ANSWER---\n"
                )
                return "reach_max_rewrite_iter"

            finally:
                state["input_tokens"].append(
                    get_num_tokens("input_tokens", cb.usage_metadata)
                )
                state["output_tokens"].append(
                    get_num_tokens("output_tokens", cb.usage_metadata)
                )


def answers_question_helper(state):
    opt_print = opt_print_factory(state["verbose"])
    opt_print(
        "---GRADE GENERATION vs QUESTION---",
    )

    question = state["question"]
    generation = state["generation"]

    answer_grade = (
        "yes"
        if state["answer_grader"]["skip"]
        else state["answer_grader"]["runnable"]
        .invoke({"question": question, "generation": generation})
        .binary_score
    )

    if answer_grade == "yes":
        opt_print(
            "---DECISION: GENERATION ADDRESSES QUESTION---\n",
        )
        return "useful"

    if state["rewrite_iteration_count"] < state["max_rewrite_iterations"]:
        opt_print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---\n")
        return "not useful"

    opt_print(
        "---DECISION: GENERATION DOES NOT ADDRESS QUESTION AND MAX REWRITE ITERATIONS REACH, CANNOT ANSWER---\n"
    )
    return "reach_max_rewrite_iter"
