from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from query_verse.db.config import SessionLocal, init_db
from query_verse.agents.rag_agent import RAGAgent
from query_verse.chat.schemas import QueryVerseInputQuery
from langchain_core.messages import HumanMessage
import time


app = FastAPI()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.on_event("startup")
# def startup_event():
#     init_db()

rag_agent = RAGAgent()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App"}

@app.post("/v1/query-verse-agent")
def query_verse(req: QueryVerseInputQuery):
    query, thread_id = req.query, req.thread_id
    
    t0 = time.time()
    res = rag_agent.agent.invoke({"question": query, "messages": HumanMessage(content=query, name="user:admin")})
    t1 = round(time.time() - t0, 2)
    return {"agent_time": t1, "agent_response": res}
    