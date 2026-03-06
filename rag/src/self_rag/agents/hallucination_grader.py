"""
Hallucination grader agent for Self-RAG workflow.

Grades whether the generated answer is grounded in the retrieved documents.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .utils import get_llm


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def create_hallucination_grader(llm_config):
    llm = get_llm(llm_config)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """
        You are a grader assessing whether an LLM generation for a given question is supported by a set of retrieved facts. \n 
        Check that the LLM generation does not contain information that do not exist in the set of facts, but the generation can contain common knowledge. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer for the question is supported by the set of facts.
    """

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n  Question: {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader
