from langchain_core.messages import AIMessage

from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langgraph.graph.state import StateGraph, END

from langchain_community.tools.tavily_search import TavilySearchResults

from query_verse.chains.document_grader import create_document_grader_chain
from query_verse.chains.rag_generation import create_rag_writer
from query_verse.chains.question_rewriter import create_query_transformer_chain
from query_verse.chains.answer_grader import create_answer_grader_chain
from query_verse.chains.hallucination_grader import create_hallucination_grader_chain


class RagState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    curr_date: str
    generation: str
    documents: str
    web_search_results: str


class RAGAgent:
    llm = ChatOpenAI(model="gpt-4o")
    lighter_llm = ChatOpenAI(model="gpt-4o-mini")
    embedding_model = OpenAIEmbeddings()
    index_name = "query-verse"
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    web_search_tool = TavilySearchResults(max_results=2)

    def __init__(self):
        self.retrieval_grader = create_document_grader_chain(model=self.llm)
        self.writer = create_rag_writer(model=self.llm)
        self.transform_query_chain = create_query_transformer_chain(
            model=self.lighter_llm
        )
        self.answer_grader = create_answer_grader_chain(model=self.llm)
        self.hallucination_grader = create_hallucination_grader_chain(model=self.llm)

        self.re_writer_counter = 1

        workflow = StateGraph(RagState)

        workflow.add_node("retriever", self.retrieve)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieval_grader", self.grade_documents)
        workflow.add_node("writer", self.generate)
        workflow.add_node("query_transformer", self.query_transformation)
        workflow.add_node("final_message", self.final_message)

        workflow.set_entry_point("retriever")

        workflow.add_edge("retriever", "retrieval_grader")
        workflow.add_edge("web_search", "retrieval_grader")

        workflow.add_conditional_edges(
            "retrieval_grader",
            self.context_relevance,
            {
                "transform_query": "query_transformer",
                "web_search": "web_search",
                "generate": "writer",
            },
        )

        workflow.add_edge("query_transformer", "retriever")
        workflow.add_conditional_edges(
            "writer",
            self.groundedness_v_answer_relevance,
            {
                "not supported": "writer",  # if not grounded or hallucination
                "useful": "final_message",  # when addresses the question
                "not useful": "query_transformer",  # when does not address the question
            },
        )
        workflow.add_edge("final_message", END)
        
        self.agent = workflow.compile()

    ## Utility functions
    # Post-processing
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def final_message(self, state: RagState):
        return {"messages": [AIMessage(content=state["generation"], name="RAG Agent")]}

    ## Nodes definition
    def retrieve(self, state: RagState):
        print("----Retrieving Documents----")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents}

    def web_search(self, state: RagState):
        print("----Making Web Search----")

        question = state["question"]
        search_result = self.web_search_tool.invoke({"query": question})

        docs = [
            Document(
                page_content=r["content"],
                metadata={"source": r["url"], "mode": "websearch"},
            )
            for r in search_result
        ]

        documents = state["documents"] + docs
        return {"web_search_results": docs, "documents": documents}

    def grade_documents(self, state: RagState):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []

        for doc in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def generate(self, state: RagState):
        print("---GENERATE---")

        question = state["question"]
        filtered_documents = state["documents"]

        # RAG generation
        generation = self.writer.invoke(
            {"context": self.format_docs(filtered_documents), "question": question}
        )
        return {"question": question, "generation": generation}

    def query_transformation(self, state: RagState):
        print("---TRANSFORM QUERY---")
        question = state["question"]

        # Re-write question
        better_question = self.transform_query_chain.invoke({"question": question})
        return {"question": better_question}

    ## Edges
    def context_relevance(self, state: RagState):
        """Context Relevance is one of the three triads for RAG Evaluation which deals with retrieved documents relevance. If None of the retrieved documents is relevant then we re-write the user's query and try to retrieve relevant documents"""
        filtered_documents = state["documents"]
        print("filtered documents")
        print(filtered_documents)

        if not filtered_documents:
            if self.re_writer_counter >= 1:
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                self.re_writer_counter -= 1
                return "transform_query"
            else:
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION AGAIN, WEB SEARCH---"
                )
                return "web_search"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def groundedness_v_answer_relevance(self, state: RagState):
        """Groundedness and Answer Relevance are the remaining two RAG evaluation triads."""
        print("---CHECK HALLUCINATIONS OR GROUNDEDNESS---")

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": self.format_docs(documents), "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
