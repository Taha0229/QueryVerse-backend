import os
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from query_verse.chat.graph import SupervisorAgent
from query_verse.chat.schemas import QueryVerseInputQuery, AddConversationHistory
from query_verse.db.conversation_history_manager import ConversationHistoryManager
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage


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

MONGODB_URI = f'{os.getenv("MONGO_URI")}'
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)
query_verse = SupervisorAgent(checkpointer=checkpointer)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App"}


@app.post("/v1/query-verse-agent")
def inference(req: QueryVerseInputQuery):
    query, thread_id = req.query, req.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    res = query_verse.agent.invoke(
        {"question": query, "messages": HumanMessage(content=query, name="user_admin")},
        config,
    )
    t1 = round(time.time() - t0, 2)
    return {"agent_time": t1, "agent_response": res["messages"][-1].content}

# messages in a conversation
@app.get("/v1/get-message-history")
def get_message_history(thread_id: str = Query(...)):
    agent = query_verse.agent
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config).values
    
    conversation_pairs = []
    pending_human_message = None
    counter = 1
    try:
        for  msg in state["messages"]:
            if isinstance(msg, HumanMessage) and msg.content:
                pending_human_message = msg.content
            elif isinstance(msg, AIMessage) and msg.content and pending_human_message:
                conversation_pairs.append({
                    "id": counter,
                    "userMessage": pending_human_message,
                    "agentMessage": msg.content
                })
                pending_human_message = None
                counter += 1
                
    except Exception as e:
        print(f"An error occurred: {e}")
                    
    return conversation_pairs

# get all conversations/chats
@app.get("/v1/get-conversation-history")
def get_message_history():
    manager = ConversationHistoryManager()
    all_chats = manager.get_conversation_history().to_dict(orient="records")
    return all_chats

@app.post("/v1/add-conversation-history")
def get_message_history(req: AddConversationHistory):
    thread_id, chat = req.thread_id, req.chat
    manager = ConversationHistoryManager()
    chat_entry = manager.add_conversation(thread_id, chat)
    if chat_entry:
        return {"message": "successfully added conversation", "chat": chat_entry}
    else:
        return {"message": "Something went wrong"}
    