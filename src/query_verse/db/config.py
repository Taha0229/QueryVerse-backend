from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from query_verse.models.config import Base
from query_verse.config import BASE_DIR

DATABASE_URL = f"sqlite:///{BASE_DIR}/test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)


