import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional, TypedDict, List, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from query_verse.chains.supervisor import create_supervisor
from query_verse.agents.rag import RAGAgent
from query_verse.chains.conversational import create_conversational_chain

load_dotenv(find_dotenv())


class AgentState(TypedDict):
    messages: Annotated[
        list, add_messages
    ]  # after completion messages[-1] has to be AI response and messages[-2] has to be HumanMessage
    question: str
    curr_date: str


class Graph:
    llm = ChatOpenAI(model="gpt-4o")
    lighter_llm = ChatOpenAI(model="gpt-4o-mini")

    def __init__(self, checkpointer: Optional[str] = None):
        self.supervisor = create_supervisor(model=self.llm)
        self.rag = RAGAgent()
        self.conversation_chain = create_conversational_chain(model=self.lighter_llm)

        graph = StateGraph(AgentState)

        graph.add_node("rag_agent", self.rag_agent)
        graph.add_node("conversational_agent", self.conversational_agent)
        graph.add_node("sql_agent", self.sql_agent)

        graph.add_conditional_edges(
            START,
            self.supervise,
            {
                "SQL Agent": "sql_agent",
                "RAG Agent": "rag_agent",
                "Conversational Agent": "conversational_agent",
            },
        )

        graph.add_edge("rag_agent", END)
        graph.add_edge("conversational_agent", END)
        graph.add_edge("sql_agent", END)

        self.graph = graph.compile()

    def supervise(self, state: AgentState):
        print("---- SUPERVISOR -----")
        res = self.supervisor.invoke({"question": state["question"]})
        print(res)
        return res.next

    def rag_agent(self, state: AgentState):
        print("---- NAVIGATING TO RAG ----")
        return self.rag.agent.invoke(
            state
        )  # updating the Graph's state with overlapping keys in RAG state

    def conversational_agent(self, state: AgentState):
        print("---- CONVERSATIONAL AGENT -----")
        conv_res = self.conversation_chain.invoke({"question": state["question"]})
        return {"messages": [AIMessage(content=conv_res, name="Conversational Agent")]}

    def sql_agent(self, state: AgentState):
        print("---- SQL Agent ----")
        return {
            "messages": [
                AIMessage(content="SQL Agent is under construction", name="SQL Agent")
            ]
        }


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
