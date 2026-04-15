from pydantic import BaseModel, Field

from enum import Enum

class Choice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class MCQAnswer(BaseModel):
    """the answer to a multiple choice question"""
    answer: Choice = Field(..., description= "the selected answer choice, must be one of 'A', 'B', 'C', or 'D'.")

