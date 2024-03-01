import argparse
import shutil
from constants import PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from utils import get_embeddings



def main(args):
        
    db = Chroma(embedding_function = get_embeddings(), persist_directory=PERSIST_DIRECTORY)
    vec_counts = db._collection.count()
    
    if args.delete_db:
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f'Delete whole DB {PERSIST_DIRECTORY}')
    
    if args.delete_text:
        id = args.source_path
        db.delete(ids = id)
        print(f'Delete {db._collection.count() - vec_counts} file')

    if args.update_text:
        id = args.source_path
        print("Before Update: ",db._collection.get(ids = id))
        doc = Document(page_content=args.update_text, metadata={"source": args.source_path})
        db.update_document(id, doc)
        print("After Update: ",db._collection.get(ids = id))
        print("successfully updated the file!")

    if args.add_text:
        id = args.source_path
        db._collection.add(ids= id, documents = args.add)
        print(f"Successfully added {db._collection.count() - vec_counts} file!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB management")

    # Add arguments
    parser.add_argument('--source_path', '-s', type=str, help='file source path')
    parser.add_argument('--delete_db', action='store_true', help='delete whole DB')
    parser.add_argument('--delete_text', action='store_true', help='delete document in DB')
    parser.add_argument('--add_text', type=str, help='add new chunk in DB')
    parser.add_argument('--update_text', type=str, help='update document in DB')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)

