from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .utils import get_llm


class GradeCorrectness(BaseModel):
    """Binary score to assess the correctness of the LLM generated answer."""

    binary_score: str = Field(
        description="LLM generated answer is correct, 'yes' or 'no'"
    )


def create_correctness_grader(llm_config: Dict):
    llm = get_llm(llm_config)
    structured_llm_grader = llm.with_structured_output(GradeCorrectness)

    system = """
        You are a grader assessing whether an LLM generated answer matches with the expected / correct answer for a given question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the LLM answer matches with the correct answer.
    """

    correctness_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Question: {question} \n\n LLM generated answer: {llm_answer} \n\n Correct answer: {correct_answer}",
            ),
        ]
    )

    correctness_grader = correctness_prompt | structured_llm_grader
    return correctness_grader
