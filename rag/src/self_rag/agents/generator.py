from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .utils import get_llm


def create_generator(llm_config: Dict):
    llm = get_llm(llm_config)
    system = """
        You are an assistant for question-answering tasks. \n
        Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Use three sentences maximum and keep the answer concise.
    """

    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Context: \n\n {context} \n\n Question: {question} \n\n Answer:"),
        ]
    )

    generator = generator_prompt | llm | StrOutputParser()
    return generator
