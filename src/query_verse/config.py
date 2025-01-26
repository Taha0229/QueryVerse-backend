from pathlib import Path
import os
from typing import List
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARENT_DIR = BASE_DIR.parent
ALLOWED_HOSTS = ['*']