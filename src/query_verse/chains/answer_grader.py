from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from query_verse.chat.parsers import GradeAnswer


def create_answer_grader_chain(
    model: Optional[LanguageModelLike] = ChatOpenAI(model="gpt-4o"),
):
   

    system_prompt = """You are a grader assessing whether an answer addresses / resolves a question.
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


    # Retriever Answer Grader chain
    answer_grader = answer_prompt | model.with_structured_output(GradeAnswer)
    
    return answer_grader
