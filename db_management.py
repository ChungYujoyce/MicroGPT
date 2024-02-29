import json
import argparse
from constants import PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from utils import get_embeddings
import uuid


def main(args):
    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)
        
    db = Chroma(embedding_function = get_embeddings(), persist_directory=PERSIST_DIRECTORY)
    vec_counts = db._collection.count()

    if args.delete:
        id = mapping[args.source_path]
        db.delete(ids = id)
        del mapping[args.source_path]
        print(f'Delete {db._collection.count() - vec_counts} file')

    if args.update:
        id = mapping[args.source_path]
        print("Before Update: ",db._collection.get(ids = id))
        doc = Document(page_content=args.update, metadata={"source": args.source_path})
        db.update_document(id, doc)
        print("After Update: ",db._collection.get(ids = id))
        print("successfully updated the file!")

    if args.add:
        id = str(uuid.uuid4())
        mapping[str(args.source_path)] = id
        db._collection.add(ids= id, documents = args.add)
        print(f"Successfully added {db._collection.count() - vec_counts} file!")
    if args.add or args.delete:
        with open(f'{PERSIST_DIRECTORY}/mapping.json', 'w') as f:
            json.dump(mapping, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB management")

    # Add arguments
    parser.add_argument('--mapping_path', '-m', type=str, default=f'{PERSIST_DIRECTORY}/mapping.json', help='mapping.json path')
    parser.add_argument('--source_path', '-s', type=str, help='file source path')
    parser.add_argument('--delete', '-d', action='store_true', help='delete document in DB')
    parser.add_argument('--update', '-u', type=str, help='update document in DB')
    parser.add_argument('--add', '-a', type=str, help='add new document in DB')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)

