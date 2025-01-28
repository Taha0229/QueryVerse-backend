from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from typing import Optional, TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate



from query_verse.agents.rag import RAGAgent
from query_verse.agents.sql import SQLAgent
from query_verse.chains.conversational import create_conversational_chain

load_dotenv(find_dotenv())
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    curr_date: str

class SupervisorAgent:
    rag = RAGAgent()
    sql = SQLAgent()
    conv_agent = create_conversational_chain()
    
    def __init__(self, checkpointer: Optional[str] = None):
        workflow = StateGraph(SupervisorState)

        self.tools = [self.rag_agent, self.sql_agent, self.conversational_agent]
        self.tools_node = ToolNode(self.tools)

        workflow.add_node("supervisor", self.supervise)
        workflow.add_node("tools", self.tools_node)
        workflow.add_edge("tools", "supervisor")
        workflow.add_conditional_edges(
            "supervisor", self.should_continue, ["tools", END]
        )
        workflow.set_entry_point("supervisor")



        self.agent = workflow.compile(checkpointer=checkpointer)

    def should_continue(self, state: SupervisorState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def supervise(self, state: SupervisorState):

        system_prompt = """
        Context:
        1. You are a supervisor, responsible for understanding the user's intend, and facilitating the collaborations between agents. 
        You have been paired with three agents to carry out tasks around Orders, Users and Products.
        names of the agents: {members}

        2. Worker Details and Capabilities:
        - RAG Agent: This Retrieval-Augmented Generation Agent is paired with a vector store containing comprehensive details about various products.  
        - SQL Agent: This agent has access to the Orders, Users, and Products tables. It can answer queries specifically using these tables. The Products table includes limited inventory details: product ID, product name, and product price only.  
        - Conversation Agent: A general-purpose agent designed for casual conversations with users.  

        3. Key Considerations:  
        - For detailed product information beyond inventory (e.g., specifications or descriptions), you may use the RAG Agent.  
        - Use the SQL Agent for queries involving inventory, user data, order information, or any relational data.  
        - The Conversation Agent should handle casual or open-ended interactions.   

        Instructions:  
        1. DO NOT use your trained knowledge to answer a question.
        2. If you think only one agent is enough to fulfill the user's question then invoke only one agent.
        3. If collaboration among the agents are required, you may invoke the agents with the part of the user's question that they can handle. 
        4. The agents will provide their response once they are done.
        5. Continue delegating tasks until the request is resolved. 
        6. When using conversational agent just pass the user's question as it is without any modification
        """

        _members = ["rag_agent", "sql_agent", "conversational_agent"]
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{messages}"),
            ]
        ).partial(members=_members)
        _model = ChatOpenAI(model="gpt-4o")

        supervisor = route_prompt | _model.bind_tools(
            self.tools, parallel_tool_calls=False
        )
        res = supervisor.invoke({"messages": state["messages"]})
        return {"messages": [res]}

    @staticmethod
    @tool
    def rag_agent(
        question: Annotated[
            str, "The query or the part of the query that the rag_agent can handle."
        ]
    ):
        """The rag_agent can provide comprehensive details on all kind of products which are in the Products table as well as which are not; using retrieval augmented generation paired with web searching capabilities"""
        print("rag_agent called")
        res = SupervisorAgent.rag.agent.invoke({"question": question})
        return res["generation"]

    @staticmethod
    @tool
    def sql_agent(
        question: Annotated[
            str, "The query or the part of the query that the sql_agent can handle."
        ]
    ):
        """The sql_agent has direct access of SQL database which has Orders, Products and Users tables. The Products table includes limited inventory details: product ID, product name, and product price only, if further details on products are required, then you may call rag_agent"""
        print("sql_agent called")
        res = SupervisorAgent.sql.agent.invoke({"question": question, "messages": [HumanMessage(content=question)]})
        return res["messages"][-1].content
    
    @staticmethod
    @tool
    def conversational_agent(question: Annotated[
            str, "Pass the user asked question as it is without any changes."
        ]):
        """The conversational agent is responsible to respond to casual conversation and greetings"""
        
        print("conversational agent called")
        res = SupervisorAgent.conv_agent.invoke({"question": question})
        return res
    
    
    
# if __name__ == "__main__":
#     agent = SupervisorAgent()
    
#     from langchain_core.runnables.graph import MermaidDrawMethod
#     from IPython.display import Image, display

#     png_data = agent.agent.get_graph().draw_mermaid_png(
#         draw_method=MermaidDrawMethod.API,
#     )
#     with open("public/supervisor_agent.png", "wb") as f:
#         f.write(png_data)

#     print("Image saved as 'agent_graph.png'")

#     from langchain_core.messages import HumanMessage

#     
#     questions = [
#         "who has purchased the most products?",
#         "What is the price of iphone 14",
#         "What all products has Neo has purchased",
#         "Who has bought the most number of products",
#         "Hello",
#         "What is your name?",
#         "What are the key features of iphone 14",
#         "What is Legion M600",
#         "Compare between Samsung S24 and iphone 14",
#         "What are features of Google Pixel 9",
#         "What is the price of Realme 9 Pro Plus in India",
#         "Can you get me details of product with product id 5",
#         "Provide in depth details of the products that Alice has purchased",
#         "Find me insights on most ordered product",
#         "What are features of cheapest product"
#     ]    
#     questions = [
#         # "who has purchased the most products?",
#         # "What is the price of iphone 14",
#         # "What all products has Neo has purchased",
#         # "Who has bought the most number of products",
#         "Can you get me details of product with product id 5",
#         "Provide in depth details of the products that Alice has purchased",
#         "Find me insights on most ordered product",
#         "What are features of cheapest product"
#     ]
    
#     from query_verse.config import BASE_DIR

#     with open(f"{BASE_DIR}/tests/QueryVerse/output.txt", "a", encoding="utf-8") as file:
#         for ind, question in enumerate(questions):
#             print(f"/*--- Performing: {question} ---*/")
#             response = agent.agent.invoke({"question": question, "messages": [HumanMessage(content=question)]})
#             file.writelines(f"{ind+1}. {question}\n{response['messages'][-1].content}\n\n")