"""
Answer grader agent for Self-RAG workflow.

Grades whether the generated answer addresses the original question.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .utils import get_llm


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def create_answer_grader(llm_config):
    llm = get_llm(llm_config)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """
        You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    return answer_grader
