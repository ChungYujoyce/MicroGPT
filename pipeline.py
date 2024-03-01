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

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    PARSED_DIRECTORY,
)

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.command()
@click.option(
    "--update",
    default=0,
    help="update the database",
)
            

def main(device_type, update):
    
    if update:
        ROOT = os.path.dirname(os.path.realpath(__file__))
        parse_dir = f"{ROOT}/SOURCE_TMP"
        source_dir = f"{ROOT}/PARSED_TMP"
    else:
        parse_dir = PARSED_DIRECTORY
        source_dir = SOURCE_DIRECTORY
        
    Path(parse_dir).mkdir(parents=True, exist_ok=True)

    doc_list = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_name = os.path.splitext(file)[0]
            source_file_path = os.path.join(root, file)

            table_dict, text_dict = pdf_prep(parse_dir, file_name, source_file_path)

            paragraph_path = f'{parse_dir}/{file_name}/paragraphs'
            Path(paragraph_path).mkdir(parents=True, exist_ok=True)
            doc_list += text_to_chunk(table_dict, text_dict, paragraph_path)
            
    doc_sources = [d.metadata['source'] for d in doc_list]
    doc_ids = [str(uuid.uuid4()) for _ in range(len(doc_sources))]
    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    db = Chroma.from_documents(
        doc_list,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
        ids=doc_ids,
    )
    
    with open(f'{PERSIST_DIRECTORY}/mapping.json', 'w') as f:
        json.dump({source:_id  for _id, source in zip(doc_ids, doc_sources)}, f, indent=4)

    # move the contents in tmp folders to original one and delete after the updates
    if update:
        shutil.copyfile(source_dir, SOURCE_DIRECTORY)
        shutil.copyfile(parse_dir, PARSED_DIRECTORY)
        try:
            shutil.rmtree(parse_dir)
            shutil.rmtree(source_dir)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
        
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()





