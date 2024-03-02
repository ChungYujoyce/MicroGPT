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
        ids = args.ids
        db.delete(ids = ids)
        print(f'Delete {db._collection.count() - vec_counts} file')

    if args.update_text:
        ids = args.ids
        print("Before Update: ",db._collection.get(ids = ids))
        doc = Document(page_content=args.update_text, metadata={"source": args.source_path})
        db.update_document(ids, doc)
        print("After Update: ",db._collection.get(ids = ids))
        print("successfully updated the file!")

    if args.add_text:
        ids = args.ids
        db._collection.add(ids= ids, documents = args.add_text)
        print(f"Successfully added {db._collection.count() - vec_counts} file!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB management")

    # Add arguments
    parser.add_argument('--source_path', '-s', type=str, help='file source path')
    parser.add_argument('--delete_db', action='store_true', help='delete whole DB')
    parser.add_argument('--delete_text', action='store_true', help='delete document in DB')
    parser.add_argument('--add_text', type=str, help='add new chunk in DB')
    parser.add_argument('--update_text', type=str, help='update document in DB')
    parser.add_argument('--ids', type=list, help='designate id')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)

