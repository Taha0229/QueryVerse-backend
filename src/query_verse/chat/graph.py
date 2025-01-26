import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional, TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv(find_dotenv())


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    date: str
    topic: str


class Graph:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def __init__(self, checkpointer: Optional[str] = None):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self.node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        
        self.graph = graph.compile()

    def node(self, state: AgentState):
        system_prompt = "Tell me a joke on the following topic: {topic}"
        
        _prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                
            ]
        )

        chain = _prompt | self.model

        response = chain.invoke({"topic": state["topic"]})
        # print("="*100)
        # print(response)
        # print("="*100)
        return {"messages": [AIMessage(content=response.content, name="agent")]}


bot = Graph()
res = bot.graph.invoke({"topic": "Mr bean"})
print(res)