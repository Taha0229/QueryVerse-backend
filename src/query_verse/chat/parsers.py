from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# Document Grader Parser
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    reason: str = Field(description="Why do you think the document is relevant or not")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class SupervisorParser(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["RAG Agent", "SQL Agent", "Conversational Agent", "FINISH"]
