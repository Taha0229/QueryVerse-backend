from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from query_verse.chat.parsers import GradeDocuments


def create_document_grader_chain(
    model: Optional[LanguageModelLike] = None,
):
    _model = ChatOpenAI(model="gpt-4o")
    
    if model:
        _model = model    

    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
    To do so, first understand the user's question. Mark the document and relevant ONLY if you are confident.
    If the document contains semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Also, specify why you marked the document as relevant or not relevant."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )


    # Retriever grader chain
    retrieval_grader = grade_prompt | _model.with_structured_output(GradeDocuments)
    
    return retrieval_grader
