import os
import time
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from sqlalchemy.orm import Session
from query_verse.db.config import SessionLocal, init_db
from query_verse.chat.graph import Graph
from query_verse.chat.schemas import QueryVerseInputQuery
from langchain_core.messages import HumanMessage
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(middleware=middleware)

MONGODB_URI = f"{os.getenv("MONGO_URI")}"
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)
query_verse = Graph(checkpointer=checkpointer)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App"}


@app.post("/v1/query-verse-agent")
def inference(req: QueryVerseInputQuery):
    query, thread_id = req.query, req.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    res = query_verse.graph.invoke(
        {"question": query, "messages": HumanMessage(content=query, name="user_admin")},
        config,
    )
    t1 = round(time.time() - t0, 2)
    return {"agent_time": t1, "agent_response": res["messages"][-1].content}
