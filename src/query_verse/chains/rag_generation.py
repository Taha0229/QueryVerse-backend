from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_rag_writer(
    model: Optional[LanguageModelLike] = ChatOpenAI(model="gpt-4o"),
):

    system_prompt = """
    Task: You are a writer assistant for question-answering task. You are provided with various information sources to answer the question.
    
    Instructions: 
    1. Generate comprehensive answers, you can use plain text as well as markdown to stand out your answer.  
    2. DO NOT fabricate an answer, stay grounded with the provided context.
    3. Optimally make you use of the provided context.

    Question: {question}

    Context: {context}

    Answer: """
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )

    # Retriever grader chain
    writer = writer_prompt | model | StrOutputParser()

    return writer
