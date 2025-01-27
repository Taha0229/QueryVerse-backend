from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from typing import Optional, TypedDict, Annotated
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate



from query_verse.chains.supervisor import create_supervisor
from query_verse.agents.rag import RAGAgent
from query_verse.agents.sql import SQLAgent
from query_verse.chains.conversational import create_conversational_chain

load_dotenv(find_dotenv())


# class AgentState(TypedDict):
#     messages: Annotated[
#         list, add_messages
#     ]  # after completion messages[-1] has to be AI response and messages[-2] has to be HumanMessage
#     question: str
#     curr_date: str


# class Graph:
#     llm = ChatOpenAI(model="gpt-4o")
#     lighter_llm = ChatOpenAI(model="gpt-4o-mini")

#     def __init__(self, checkpointer: Optional[str] = None):?
#         self.supervisor = create_supervisor(model=self.llm)
#         self.rag = RAGAgent()
#         self.sql = SQLAgent()
#         self.conversation_chain = create_conversational_chain(model=self.lighter_llm)

#         graph = StateGraph(AgentState)

#         graph.add_node("rag_agent", self.rag_agent)
#         graph.add_node("conversational_agent", self.conversational_agent)
#         graph.add_node("sql_agent", self.sql_agent)

#         graph.add_conditional_edges(
#             START,
#             self.supervise,
#             {
#                 "SQL Agent": "sql_agent",
#                 "RAG Agent": "rag_agent",
#                 "Conversational Agent": "conversational_agent",
#                 "FINISH": END
#             },
#         )

#         graph.add_edge("rag_agent", END)
#         graph.add_edge("conversational_agent", END)
#         graph.add_edge("sql_agent", END)

#         self.graph = graph.compile(checkpointer=checkpointer)

#     def supervise(self, state: AgentState):
#         print("---- SUPERVISOR -----")
#         res = self.supervisor.invoke({"question": state["question"]})
#         print(res)
#         return res.next

#     def rag_agent(self, state: AgentState):
#         print("---- NAVIGATING TO RAG ----")
#         return self.rag.agent.invoke(
#             state
#         )  # updating the Graph's state with overlapping keys in RAG state -> The messages in RAG is handled in a way to just return final AI message

#     def conversational_agent(self, state: AgentState):
#         print("---- CONVERSATIONAL AGENT -----")
#         conv_res = self.conversation_chain.invoke({"question": state["question"]})
#         return {"messages": [AIMessage(content=conv_res, name="conversational_agent")]}

#     def sql_agent(self, state: AgentState):
#         print("---- SQL Agent ----")
#         res = self.sql.agent.invoke(state)
#         return {"messages": [res["messages"][-1]]}


# test code

# def carry_test():
#     bot = Graph()
#     start = True
#     while start:
#         inp = input("Enter your question: ")
#         res = bot.graph.invoke(
#             {"question": inp, "messages": [HumanMessage(content=inp)]}
#         )
#         print(res)
#         should_stop = input(
#             "Do you want to stop? Press Y to stop any other key to continue: "
#         )
#         if should_stop.lower() == "y":
#             start = False
#         else:
#             continue


# carry_test()


class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    curr_date: str


class PlannerAgent:
    rag = RAGAgent()
    sql = SQLAgent()
    conv_agent = create_conversational_chain()
    
    def __init__(self, checkpointer: Optional[str] = None):
        workflow = StateGraph(PlannerState)

        self.tools = [self.rag_agent, self.sql_agent, self.conversational_agent]
        self.tools_node = ToolNode(self.tools)

        workflow.add_node("planner", self.planner)
        workflow.add_node("tools", self.tools_node)
        workflow.add_edge("tools", "planner")
        workflow.add_conditional_edges(
            "planner", self.should_continue, ["tools", END]
        )
        workflow.set_entry_point("planner")



        self.agent = workflow.compile(checkpointer=checkpointer)

    def should_continue(self, state: PlannerState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def planner(self, state: PlannerState):

        system_prompt = """
        Context:
        1. You are a planner, responsible for understanding the user's intend, facilitating the collaborations between agents. 
        You have been paired with two agents to carry out tasks around Orders, Users and Products.
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
        1. If you think only one agent is enough to fulfill the user's question then invoke only one agent.
        2. If collaboration among the agents are required, you may invoke the agents with the part of the user's question that the they can handle. 
        3. The agents will provide their response once they are done.
        4. Continue delegating tasks until the request is resolved.  
        5. When using conversational agent just pass the user's question as it is without any modification
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
        """The rag_agent can provide comprehensive details on all kind of products which are in the Products table as well as which are not using retrieval augmented generation paired with web searching capabilities"""
        print("rag_agent called")
        res = PlannerAgent.rag.agent.invoke({"question": question})
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
        res = PlannerAgent.sql.agent.invoke({"question": question})
        return res["messages"][-1].content
    
    @staticmethod
    @tool
    def conversational_agent(question: Annotated[
            str, "Pass the user asked question as it is without any changes."
        ]):
        """The conversational agent is responsible to respond to casual conversation and greetings"""
        
        print("conversational agent called")
        res = PlannerAgent.conv_agent.invoke({"question": question})
        return res