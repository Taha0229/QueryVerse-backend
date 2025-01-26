from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


def create_rag_writer(
    model: Optional[LanguageModelLike] = None,
):
    _model = ChatOpenAI(model="gpt-4o")
    
    if model:
        _model = model    

    prompt = hub.pull("rlm/rag-prompt")

    # Retriever grader chain
    writer = prompt | _model |  StrOutputParser()
    
    return writer
