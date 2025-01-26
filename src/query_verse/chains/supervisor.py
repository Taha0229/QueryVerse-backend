### Router

from typing import Literal, TypedDict, List, Optional, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from query_verse.chat.parsers import SupervisorParser

        
def create_supervisor(model: Optional[LanguageModelLike] = None, members: List[str]= None):
    _model = ChatOpenAI(model="gpt-4o")
    if model:
        _model = model   
        
    _members = ["RAG Agent", "SQL Agent", "Conversational Agent"]
    if members:
        _members = members
    
    system_prompt = """
        Context:
        1. You are a supervisor managing a conversation between the following workers: {members}.  

        2. Worker Details and Capabilities:
        - RAG Agent: This Retrieval-Augmented Generation Agent is paired with a vector store containing comprehensive details about various products.  
        - SQL Agent: This agent has access to the Orders, Users, and Products tables. It can answer queries specifically using these tables. The Products table includes limited inventory details: product ID, product name, and product price only.  
        - Conversation Agent: A general-purpose agent designed for casual conversations with users.  

        3. Key Considerations:  
        - For detailed product information beyond inventory (e.g., specifications or descriptions), direct queries to the RAG Agent.  
        - Use the SQL Agent for queries involving inventory, user data, order information, or any relational data.  
        - The Conversation Agent should handle casual or open-ended interactions.  

        Instructions:  
        1. Analyze the user's request and determine the next worker to act. Workers will execute their tasks and respond with results and status updates.  
        2. Continue delegating tasks until the request is resolved.  
        3. Conclude the process by responding with FINISH when the task is fully completed.
        """

    route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        ).partial(members=_members)
        
    supervisor = route_prompt | _model.with_structured_output(SupervisorParser)
    
    return supervisor

            


