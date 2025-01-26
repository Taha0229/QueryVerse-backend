from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def create_conversational_chain( model: Optional[LanguageModelLike] = None):
    _model = ChatOpenAI(model="gpt-4o-mini")
    if model:
        _model = model   

    conversation_agent_prompt = """
        Context: Your name is QueryVerse, a multi-agent and, highly knowledgeable assistant.
        Instructions: If a user ask you anything other than casual conversation and greetings you must not reply to that
        and remind them that you are just an assistant and they should only ask something related Products, Users or Orders."""
        
    conv_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", conversation_agent_prompt),
            ("human", "{question}")
        ]
    )

    conversation_chain = conv_prompt | _model | StrOutputParser()
    return conversation_chain
