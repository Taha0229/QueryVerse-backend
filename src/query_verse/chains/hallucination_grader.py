from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from query_verse.chat.parsers import GradeHallucinations


def create_hallucination_grader_chain(
    model: Optional[LanguageModelLike] = None,
):
    _model = ChatOpenAI(model="gpt-4o")

    if model:
        _model = model

    system_prompt = """You are a grader assessing whether an LLM generation is grounded in/supported by a set of retrieved facts.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in/supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    # Hallucination grader chain
    hallucination_grader = hallucination_prompt | _model.with_structured_output(GradeHallucinations)

    return hallucination_grader
