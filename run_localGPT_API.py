import logging
import os
import shutil
import subprocess
import argparse

import torch
from flask import Flask, jsonify, request, render_template
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import VLLMOpenAI

# from langchain.embeddings import HuggingFaceEmbeddings
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from utils import clean_text
from werkzeug.utils import secure_filename

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, PARSED_DIRECTORY, MODEL_ID, MODEL_BASENAME
from db_mng import DB_Management

# API queue addition
from threading import Lock

request_lock = Lock()


if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"
    
print(DEVICE_TYPE)
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# if os.path.exists(PERSIST_DIRECTORY):
#     try:
#         shutil.rmtree(PERSIST_DIRECTORY)
#     except OSError as e:
#         print(f"Error: {e.filename} - {e.strerror}.")
# else:
#     print("The directory does not exist")

# run_langest_commands = ["python", "ingest.py"]
# if DEVICE_TYPE == "cpu":
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

# result = subprocess.run(run_langest_commands, capture_output=True)
# if result.returncode != 0:
#     raise FileNotFoundError(
#         "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
#     )

# load the vectorstore
LLM = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.7:5000/v1",
        model_name="test",
        max_tokens=512,
        temperature=0,
        model_kwargs={
            "stop": [],
        },
    )

DB = None
RETRIEVER = None
RETRIEVER_BM25 = None
QA = None
db_manager = None

def load_DB():
    global DB
    global RETRIEVER
    global RETRIEVER_BM25
    global QA
    global app
    global db_manager
    # load the vectorstore
    
    # Return document size
    k = 4
    
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    app.logger.info(f'DB size: {DB._collection.count()}')

    RETRIEVER = DB.as_retriever(search_kwargs={"k": k * 2})
    
    collections = DB.get()
    documents = [Document(page_content=c, metadata=m) for m, c in zip(collections['metadatas'], collections['documents'])]
    RETRIEVER_BM25 = BM25Retriever.from_documents(documents=documents, preprocess_func=clean_text, k=k)
    
    prompt, memory = get_prompt_template(promptTemplate_type="llama3", history=False)

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        retriever_bm25=RETRIEVER_BM25,
        return_source_documents=SHOW_SOURCES,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )
    
    db_manager = DB_Management(f'{PERSIST_DIRECTORY}/mapping.json', PERSIST_DIRECTORY)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
load_DB()


@app.route("/api/delete_source", methods=["DELETE"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})


@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_TMP"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_path = os.path.join(folder_path, filename)

        file.save(file_path)
        return "File saved successfully", 200

@app.route("/api/run_add", methods=["POST"])
def run_add():
    
    try:
        run_langest_commands = ["python", "pipeline.py", "--source_dir", "SOURCE_TMP", "--parse_dir", "PARSED_TMP"]
        
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        
        shutil.copytree("SOURCE_TMP", "SOURCE_DOCUMENTS", dirs_exist_ok=True)
        shutil.copytree("PARSED_TMP", "PARSED_DOCUMENTS", dirs_exist_ok=True)
        shutil.rmtree("SOURCE_TMP")
        shutil.rmtree("PARSED_TMP")
        
        load_DB()
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

@app.route("/api/run_delete", methods=["DELETE"])
def run_delete():

    try:
        # Cheng-Ping Love you very much.
        global request_lock  # Make sure to use the global lock instance
        _id = request.form.get("id")
        app.logger.info(_id)

        with request_lock:
            db_manager.delete_text(_id)
        
        load_DB()
        return "Script executed successfully", 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/run_update", methods=["PUT"])
def run_update():
    try:
        global request_lock  # Make sure to use the global lock instance
        # original_result is a jsonify dict
        _id, revise_result = request.form.get("id"), request.form.get("revise_result")
        app.logger.info(_id)
        app.logger.info(revise_result)
        
        with request_lock:
            db_manager.update_text(_id, revise_result.strip())
        
        load_DB()
        return "Script executed successfully", 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500
    

@app.route("/api/run_reset", methods=["POST", "DELETE"])
def run_reset():
    try:
        db_manager.delete_db()
        # result = subprocess.run(["python", "db_management.py", "--delete_db"], capture_output=True)
        
        run_langest_commands = ["python", "pipeline.py", "--source_dir", "SOURCE_TMP", "--parse_dir", "PARSED_TMP"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        
        shutil.rmtree("SOURCE_DOCUMENTS")
        shutil.rmtree("PARSED_DOCUMENTS")
        shutil.move("SOURCE_TMP", "SOURCE_DOCUMENTS")
        shutil.move("PARSED_TMP", "PARSED_DOCUMENTS")

        load_DB()
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500
    
@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    global request_lock  # Make sure to use the global lock instance
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        # Acquire the lock before processing the prompt
        with request_lock:
            # print(f'User Prompt: {user_prompt}')              
            # Get the answer from the chain
            res = QA(user_prompt)
            answer, docs = res["result"], res["source_documents"]

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }

            prompt_response_dict["Sources"] = []
            for document in docs:
                _id = document.metadata["id"]
                source = "/".join(str(document.metadata["source"]).split("/")[-3:])
                prompt_response_dict["Sources"].append((_id, source, str(document.page_content)))

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110, help="Port to run the API on. Defaults to 5110.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
        "Set to 0.0.0.0 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host=args.host, port=args.port)
