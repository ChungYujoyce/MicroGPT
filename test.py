from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from utils import get_embeddings
import json

from db_mng import DB_Management
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, PARSED_DIRECTORY, MODEL_ID, MODEL_BASENAME


db_manager = DB_Management(f'{PERSIST_DIRECTORY}/mapping.json', PERSIST_DIRECTORY)

with open('./db_result.json', "w") as f:
    mapping = db_manager.db._collection.get()
    json.dump(mapping, f, indent=4)