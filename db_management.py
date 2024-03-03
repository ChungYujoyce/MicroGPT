import argparse
import shutil
import os
import json
from constants import PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from utils import get_embeddings

## [TODO] should be a class here and initialize in run_localGPT_API.py

def main(args):
    
    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)
        
    db = Chroma(embedding_function = get_embeddings(), persist_directory=PERSIST_DIRECTORY)
    vec_counts = db._collection.count()
    # print(db._collection.get())
    
    if args.delete_db:
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f'Delete whole DB {PERSIST_DIRECTORY}')
    
    if args.delete_text:
        _id = args.id
        old_doc = db._collection.get(ids = _id)
        source_path = old_doc['metadatas'][0]["source"]
        
        # Remove local file
        os.remove(source_path)
        
        # Remove db file
        db.delete(ids = _id)
        del mapping[_id]
        print(f'Delete {vec_counts - db._collection.count()} file')

    if args.update_text:
        _id = args.id
        old_doc = db._collection.get(ids = _id)
        source_path = old_doc['metadatas'][0]["source"]
        
        # Update local file
        with open(source_path, 'w') as f:
            f.write(args.update_text)
        
        # Update db file
        new_doc = Document(page_content=args.update_text, metadata=old_doc['metadatas'][0])
        db.update_document(_id, new_doc)
        
        print("After Update: ",db._collection.get(ids = _id))
        print("Successfully updated the file!")
    
    # [TODO] add text how to support? given source_path not id
    if args.add_text:
        _id = args.id
        db._collection.add(ids= _id, documents = args.add_text)
        print(f"Successfully added {db._collection.count() - vec_counts} file!")

    with open(args.mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB management")

    # Add arguments
    parser.add_argument('--mapping_path', '-m', type=str, default=f'{PERSIST_DIRECTORY}/mapping.json', help='mapping.json path')
    parser.add_argument('--id', '-s', type=str, help='file id or source path')
    parser.add_argument('--delete_db', action='store_true', help='delete whole DB')
    parser.add_argument('--delete_text', action='store_true', help='delete document in DB')
    parser.add_argument('--add_text', type=str, help='add new chunk in DB')
    parser.add_argument('--update_text', type=str, help='update document in DB')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)

