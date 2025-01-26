from pydantic import BaseModel

class QueryVerseInputQuery(BaseModel):
    query: str
    thread_id: str