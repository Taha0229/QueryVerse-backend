from pydantic import BaseModel

class QueryVerseInputQuery(BaseModel):
    query: str
    thread_id: str
    
class AddConversationHistory(BaseModel):
    thread_id: str
    chat: str