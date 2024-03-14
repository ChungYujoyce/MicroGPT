import argparse
import shutil
import os
import json
from constants import PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from utils import get_embeddings


class DB_Management:
    def __init__(self, mapping_path, PERSIST_DIRECTORY):
        self.mapping_path = mapping_path
        self.load_mapping()
        self.persist_directory = PERSIST_DIRECTORY
        self.db = Chroma(embedding_function=get_embeddings(), persist_directory=self.persist_directory)
        self.vec_counts = self.db._collection.count()

    def delete_db(self):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f'Delete whole DB {PERSIST_DIRECTORY}')

    def delete_text(self, _id):
        before = self.vec_counts
        old_doc = self.db._collection.get(ids = _id)
        source_path = old_doc['metadatas'][0]["source"]
        os.remove(source_path) # Remove local file
        self.db.delete(ids = _id) # Remove db file
        
        del self.mapping[_id]
        self.save_mapping()
        
        print(f'Delete {before - self.db._collection.count()} file')

    def update_text(self, _id, update_text):
        old_doc = self.db._collection.get(ids = _id)
        source_path = old_doc['metadatas'][0]["source"]
        # Update local file
        with open(source_path, 'w') as f:
            f.write(update_text)
        # Update db file
        new_doc = Document(page_content = update_text, metadata = old_doc['metadatas'][0])
        self.db.update_document(_id, new_doc)

        print("After Update: ", self.db._collection.get(ids = _id))
        print("Successfully updated the file!")

    def add_text(self, _id, add_text):
        before = self.vec_counts
        self.db._collection.add(ids = _id, documents = add_text)
        print(f"Successfully added {self.db._collection.count() - before} file!")
            
    def load_mapping(self):
        with open(self.mapping_path, 'r') as f:
            self.mapping = json.load(f)

    def save_mapping(self):
        with open(self.mapping_path, 'w') as f:
            json.dump(self.mapping, f, indent=4)