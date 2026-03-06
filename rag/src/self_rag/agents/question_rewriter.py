"""
Question rewriter agent for Self-RAG workflow.

Rewrites questions to improve retrieval results.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .utils import get_llm


def create_question_rewriter(llm_config):
    llm = get_llm(llm_config)
    system = """
        You are a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
        The question must not change its purpose. \n
        In other words, the answer to the rewritten question must be the same as the answer to the original question. \n
        Only provide the revised version of the given question.
    """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n\n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter
