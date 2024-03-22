import os, logging
import click
import shutil
import torch
from pathlib import Path
from pdf_prep import pdf_prep
from chunk_prep import text_to_chunk, text_to_chunk_non_pdf
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from utils import get_embeddings
import uuid
import json
import argparse
import time
import hashlib
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
import threading
from ingest import load_single_document, load_document_batch, load_documents 

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
)


parser = argparse.ArgumentParser(description="DB management")

parser.add_argument("--device_type", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on. (Default is cuda)")
parser.add_argument('--source_dir', type=str, default="SOURCE_DOCUMENTS")
parser.add_argument('--parse_dir', type=str, default="PARSED_DOCUMENTS")
parser.add_argument("--threads", type=int, default=12)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)
args = parser.parse_args()


def main():
    def process_page(idx, file):
        file_name = file['filename']
        source_file_path = file['source_file_path']

        table_dict, text_dict = dict(), dict()
        try:
            table_dict, text_dict = pdf_prep(args.parse_dir, file_name, source_file_path)
        except:
            print(f"File {file_name} has error")

        paragraph_path = f'{args.parse_dir}/{file_name}/paragraphs'
        Path(paragraph_path).mkdir(parents=True, exist_ok=True)
        docs = text_to_chunk(table_dict, text_dict, paragraph_path, file_name)
        return docs
    
    parse_dir = args.parse_dir
    source_dir = args.source_dir
    Path(parse_dir).mkdir(parents=True, exist_ok=True)
    
    # pdf parsing
    files = [f for f in os.listdir(source_dir) if '.pdf' in f]
    files_mapping = []
    for file in files:
        files_mapping.append({
            'filename': os.path.splitext(file)[0],
            'source_file_path': os.path.join(source_dir, file)
        })

    doc_list = []
    for idx, file in enumerate(tqdm(files_mapping)):
        doc_list += process_page(idx, file)    


    # non-pdf parsing
    documents = load_documents(source_dir)
    files = [f for f in os.listdir(source_dir) if '.pdf' not in f]

    for doc, file in zip(documents, files):
        file_name = os.path.splitext(file)[0]
        text = doc.page_content
        try:
            paragraph_path = f'{parse_dir}/{file_name}/paragraphs'
            Path(paragraph_path).mkdir(parents=True, exist_ok=True)
            doc_list += text_to_chunk_non_pdf(text, paragraph_path, file_name)
        except:
            print(f"File {file_name} has error")
            
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

    
if __name__ == "__main__":
    main()





