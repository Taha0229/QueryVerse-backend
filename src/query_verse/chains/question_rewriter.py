from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from langchain_core.output_parsers import StrOutputParser



def create_query_transformer_chain(
    model: Optional[LanguageModelLike] = None,
):
    _model = ChatOpenAI(model="gpt-4o-mini")
    
    if model:
        _model = model    

    system_prompt = """You are a question re-writer that converts an input question to a better version that is optimized 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )


    # Retriever grader chain
    query_transformer_chain = re_write_prompt | _model | StrOutputParser()
    
    return query_transformer_chain
