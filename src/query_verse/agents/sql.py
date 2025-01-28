from typing import Any, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.prebuilt import ToolNode

from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from query_verse.config import BASE_DIR


class SQLState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    query: str


class SQLAgent:
    db = SQLDatabase.from_uri(f"sqlite:///{BASE_DIR}/test.db")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    lighter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def __init__(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()

        self.list_tables_tool = next(
            tool for tool in tools if tool.name == "sql_db_list_tables"
        )  # tool 1/3
        self.get_schema_tool = next(
            tool for tool in tools if tool.name == "sql_db_schema"
        )  # tool 2/3

        workflow = StateGraph(SQLState)

        workflow.set_entry_point("first_tool_call")

        workflow.add_node(
            "first_tool_call", self.first_tool_call
        )  # explicitly calling get all tables tool
        workflow.add_node(
            "list_tables_tool",
            self.create_tool_node_with_fallback([self.list_tables_tool]),
        )  # Tool Executor for the above node

        workflow.add_node(
            "model_get_schema", self.model_get_schema
        )  # LLM node, paired with get_schema_tool to pick relevant tables
        workflow.add_node(
            "get_schema_tool",
            self.create_tool_node_with_fallback([self.get_schema_tool]),
        )  # This node will execute the tool invoked by the above Node

        workflow.add_node(
            "query_gen", self.query_gen
        )  # generates first iteration of query

        workflow.add_node(
            "correct_query", self.model_check_query
        )  # checks and executes the query
        workflow.add_node(
            "execute_query",
            self.create_tool_node_with_fallback(
                [self.db_query_tool]
            ),  # Query executor tool
        )
        workflow.add_node(
            "writer", self.writer
        )  # takes the response and responds in natural language

        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_edge("query_gen", "correct_query")
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "writer")
        workflow.add_edge("writer", END)

        self.agent = workflow.compile()

    # utility methods
    def create_tool_node_with_fallback(
        self, tools: list
    ) -> RunnableWithFallbacks[Any, dict]:
        """
        Create a ToolNode with a fallback to handle errors and surface them to the agent.
        """
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )

    def handle_tool_error(self, state: SQLState) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def first_tool_call(
        self,
        state: SQLState,
    ) -> dict[str, list[AIMessage]]:  # This explicitly mimics a tool call by an LLM
        print("----Getting all tables----")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_list_tables",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }

    # Custom tool to execute query -> tool 3/3
    @staticmethod
    @tool
    def db_query_tool(query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        result = SQLAgent.db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result

    # Nodes
    def model_check_query(self, state: SQLState) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        print("----Checking the generated SQL query----")
        
        query_check_system = """You are a SQL expert with a strong attention to detail.
        Double check the SQLite query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

        You will call the appropriate tool to execute the query after running this check."""

        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", query_check_system), ("placeholder", "{messages}")]
        )
        query_check = query_check_prompt | self.llm.bind_tools(
            [self.db_query_tool], tool_choice="required"
        )

        # messages[-1] assuming that the previous message is from query writer
        return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

    def model_get_schema(self, state: SQLState):
        print("----Getting Table Schema----")
        get_schema = self.llm.bind_tools([self.get_schema_tool])
        res = get_schema.invoke(state["messages"])

        # This is returning list with one AIMessage
        return {"messages": [res]}

    def query_gen(self, state: SQLState):
        print("----Getting SQL Query----")
        query_gen_system = """
            Context: 
            You are a SQL expert with a strong attention to detail working in a multi-agent system. Your role is to generate SQLite queries based on the user's question, provided schema and instructions. 
            Below is the schema of the relevant table(s) for your reference:

            {relevant_schema}

            Additionally, a few sample rows from the relevant table(s) are provided. These are strictly for your reference and should not influence the query results.

            Instructions:
            1. Always limit your query to a maximum of 10 results unless the user explicitly specifies a different number.
            2. If a query execution results in an error, carefully rewrite the query to fix the issue and try again.
            3. If the provided schema does not contain enough information to fulfill the user's request, respond with: "I do not have enough information to answer your query."
            4. Never perform any DML (Data Manipulation Language) operations such as INSERT, UPDATE, DELETE, or DROP. Only generate SELECT queries or other safe read-only queries.
            5. Only respond with SQLite query without adding any text.
            """
        query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("human", "{question}")]
        )

        schema = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage) and msg.name == "sql_db_schema":
                schema.append(msg.content)

        generate = query_gen_prompt | self.llm
        res = generate.invoke(
            {"question": state["question"], "relevant_schema": "\n\n".join(schema)}
        )
        return {"query": res, "messages": [res]}

    def writer(self, state: SQLState):
        print("----Writing final response----")
        system_prompt = """You are writer agent working in a multi-agent system. 
        You will be provided with the user's question and its response generated by the SQL Agent. 
        Your task is to write a final answer to the user based.
        """

        writer_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{question}"), ("ai", "{response}")]
        )
        writer_chain = writer_prompt | self.lighter_llm
        res = writer_chain.invoke(
            {"question": state["question"], "response": state["messages"][-1].content}
        )
        return {"messages": [res]}


# if __name__ == "__main__":
#     sql_agent = SQLAgent()
    
#     from langchain_core.runnables.graph import MermaidDrawMethod

#     png_data = sql_agent.agent.get_graph().draw_mermaid_png(
#         draw_method=MermaidDrawMethod.API,
#     )
#     with open("public/sql_agent.png", "wb") as f:
#         f.write(png_data)

#     print("Image saved as 'sql_agent.png'")
    
#     from langchain_core.messages import HumanMessage

#     sql_agent = SQLAgent()
#     questions = [
#         "how many users are there?",
#         "who has purchased the most products?",
#         "What is the price of iphone 14",
#         "List the products purchased by Jack",
#         "What is the total order value",
#         "Which is the most expensive product that you have",
#         "How much does Harry and Issac has spent till now",
#         "What all products has leo has purchased",
#         "Who has bought the most number of products"
#     ]
#     from query_verse.config import BASE_DIR

#     with open(f"{BASE_DIR}/tests/SQL/output.txt", "a", encoding="utf-8") as file:
#         for ind, question in enumerate(questions):
#             response = sql_agent.agent.invoke({"question": question, "messages": [HumanMessage(content=question)]})
#             file.writelines(f"{ind+1}. {question}\n{response['messages'][-1].content}\n\n")

