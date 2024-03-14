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
import time
import hashlib
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
import threading


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
        table_dict, text_dict = pdf_prep(args.parse_dir, file_name, source_file_path)
        paragraph_path = f'{args.parse_dir}/{file_name}/paragraphs'
        Path(paragraph_path).mkdir(parents=True, exist_ok=True)
        docs = text_to_chunk(table_dict, text_dict, paragraph_path, file_name)
        # outputs_parallel[idx]['doc'] = docs
        return docs
    
    parse_dir = args.parse_dir
    source_dir = args.source_dir
    Path(parse_dir).mkdir(parents=True, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if '.pdf' in f]
    files_mapping = []
    for file in files:
        files_mapping.append({
            'filename': os.path.splitext(file)[0],
            'source_file_path': os.path.join(source_dir, file)
        })
    
    # start = time.time()
    doc_list = []
    for idx, file in enumerate(tqdm(files_mapping)):
        doc_list += process_page(idx, file)
    # print(f'{time.time() - start} sec')
    
# multi-process
#     docs = process_map(process_page, files_mapping, max_workers=32)
#     doc_list = [chunk for doc in docs for chunk in doc]

# multi-thread
#     start = time.time()
#     doc_list = []
#     threads = []
#     outputs_parallel = [{} for _ in range(len(files_mapping))]
#     for idx, file in tqdm(enumerate(files_mapping), total=len(files_mapping)):
#         thread = threading.Thread(
#             target=process_page,
#             kwargs=dict(
#                 idx=idx,
#                 file=file,
#             ),
#         )
#         thread.start()
#         threads.append(thread)
#         if len(threads) == args.threads:
#             for thread in threads:
#                 thread.join()
#             threads = []
#             for computed_idx in range(idx - args.threads + 1, idx + 1):
#                 doc_list += outputs_parallel[computed_idx]['doc']
#     # collecting the final batch
#     for thread in threads:
#         thread.join()
        
#     for computed_idx in range(idx - len(threads) + 1, idx + 1):
#          doc_list += outputs_parallel[computed_idx]['doc']
#     print(f'{time.time() - start} sec')
    
            
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
    main()





