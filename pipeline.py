import os, logging
import click
import shutil
import torch
from pathlib import Path
from pdf_prep import pdf_prep
from chunk_prep import text_to_chunk
from langchain.vectorstores import Chroma
from utils import get_embeddings
import uuid
import json
import argparse
import hashlib

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
)

            

def main(args):
        
    parse_dir = args.parse_dir
    source_dir = args.source_dir
    
    Path(parse_dir).mkdir(parents=True, exist_ok=True)

    doc_list = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_name = os.path.splitext(file)[0]
            source_file_path = os.path.join(root, file)

            table_dict, text_dict = pdf_prep(parse_dir, file_name, source_file_path)

            paragraph_path = f'{parse_dir}/{file_name}/paragraphs'
            Path(paragraph_path).mkdir(parents=True, exist_ok=True)
            doc_list += text_to_chunk(table_dict, text_dict, paragraph_path, file_name)
    
    doc_ids = []
    doc_sources = []
    for d in doc_list:
        d.metadata['id'] = hashlib.sha256(d.metadata['source'].encode()).hexdigest()
        doc_ids.append(d.metadata['id'])
        doc_sources.append(d.metadata['source'])
        
    embeddings = get_embeddings(args.device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    db = Chroma.from_documents(
        doc_list,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
        ids=doc_ids,
    )
    print(db._collection.count())
    
    with open(f'{PERSIST_DIRECTORY}/mapping.json', 'w') as f:
        json.dump({_id:source for _id, source in zip(doc_ids, doc_sources)}, f, indent=4)

    # move the contents in tmp folders to original one and delete after the updates
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB management")

    parser.add_argument("--device_type", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on. (Default is cuda)")
    parser.add_argument('--source_dir', type=str, default="SOURCE_DOCUMENTS")
    parser.add_argument('--parse_dir', type=str, default="PARSED_DOCUMENTS")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    args = parser.parse_args()
    main(args)





