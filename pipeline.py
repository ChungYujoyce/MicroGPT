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


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        print("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(paths: list[str]) -> list[Document]:
    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))
    return docs


def main():
    # def process_page(idx, file):
    #     file_name = file['filename']
    #     source_file_path = file['source_file_path']

    #     table_dict, text_dict = dict(), dict()
    #     try:
    #         table_dict, text_dict = pdf_prep(args.parse_dir, file_name, source_file_path)
    #     except:
    #         print(f"File {file_name} has error")

    #     paragraph_path = f'{args.parse_dir}/{file_name}/paragraphs'
    #     Path(paragraph_path).mkdir(parents=True, exist_ok=True)
    #     docs = text_to_chunk(table_dict, text_dict, paragraph_path, file_name)
    #     return docs
    
    parse_dir = args.parse_dir
    source_dir = args.source_dir
    Path(parse_dir).mkdir(parents=True, exist_ok=True)

    paths, files = [], []
    doc_list = []
    for file in os.listdir(source_dir):
        file_name = os.path.splitext(file)[0]
        file_extension = os.path.splitext(file)[1]
        source_file_path = os.path.join(source_dir, file)
        # pdf parsing
        if file_extension == '.pdf':
            table_dict, text_dict = dict(), dict()
            try:
                table_dict, text_dict = pdf_prep(parse_dir, file_name, source_file_path)
            except:
                print(f"File {file_name} has error")

            paragraph_path = f'{args.parse_dir}/{file_name}/paragraphs'
            Path(paragraph_path).mkdir(parents=True, exist_ok=True)
            doc_list += text_to_chunk(table_dict, text_dict, paragraph_path, file_name)
        else:
            # add non-pdf paths for later process
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)
                files.append(file_name)
            else:
                print(f"{file_extension} file type not support")

    # non-pdf parsing
    if len(paths) > 0:
        documents = load_documents(paths)
        for doc, file_name in zip(documents, files):
            text = doc.page_content
            try:
                paragraph_path = f'{parse_dir}/{file_name}/paragraphs'
                Path(paragraph_path).mkdir(parents=True, exist_ok=True)
                doc_list += text_to_chunk_non_pdf(text, paragraph_path, file_name)
            except:
                print(f"File {file_name} has error")

    # pdf parsing
    # files = [f for f in os.listdir(source_dir) if '.pdf' in f]
    # files_mapping = []
    # for file in files:
    #     files_mapping.append({
    #         'filename': os.path.splitext(file)[0],
    #         'source_file_path': os.path.join(source_dir, file)
    #     })

    # doc_list = []
    # for idx, file in enumerate(tqdm(files_mapping)):
    #     doc_list += process_page(idx, file)

            
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
    
    try:
        file_path = f'{PERSIST_DIRECTORY}/mapping.json'
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                mapping = json.load(f)
            doc_sources = [doc.replace("PARSED_TMP", "PARSED_DOCUMENTS") for doc in doc_sources if "PARSED_TMP" in doc]
            mapping.update({_id:source for _id, source in zip(doc_ids, doc_sources)})
        else:
            mapping = {_id:source for _id, source in zip(doc_ids, doc_sources)}
        with open(file_path, "w") as f:
            json.dump(mapping, f, indent=4)
            
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
    
if __name__ == "__main__":
    main()