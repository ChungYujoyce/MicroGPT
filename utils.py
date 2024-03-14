import pdfplumber
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import os
import csv
from datetime import datetime
from constants import EMBEDDING_MODEL_NAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


#Parse PDFs excluding tables.
def extract_text_without_tables(p, page_idx):
    try:
        ts = {"vertical_strategy": "text","horizontal_strategy": "text", "min_words_vertical": 3, "min_words_horizontal": 18, "text_tolerance": 3}
        p.to_image().debug_tablefinder(ts).save('Out.jpg')
        # import pdb
        # pdb.set_trace()
        
        # Get the bounding boxes of the tables on the page.
        bboxes = [table.bbox for table in p.find_tables(table_settings=ts)]
        table_texts, raw_texts = "", ""
        if len(bboxes) > 0:
            head = 0
            for idx, __bbox in enumerate(bboxes):
                x0, top, x1, bottom = __bbox
                table_texts += p.crop((0, head, p.width, top), relative=False, strict=False).extract_text()
                table_texts += f'<|page_{page_idx}_table_{idx+1}|>'
                head = bottom
            raw_texts = p.crop((0, head, p.width, p.height), relative=False, strict=False).extract_text()
        else:
            raw_texts = p.extract_text()
    except:
        v_lines, h_lines = [], []
        if len(p.lines) > 0:
            h_pnt_cnts = Counter([(line['x0'], line['x1']) for line in p.lines if line['height'] == 0])
            v_lines = [index for k, v in h_pnt_cnts.items() if v > 1 for index in k]
            v_pnt_cnts = Counter([(line['y0'], line['y1']) for line in p.lines if line['width'] == 0])
            h_lines = [index for k, v in v_pnt_cnts.items() if v > 1 for index in k]
            
        ts = {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "explicit_vertical_lines": v_lines,
            "explicit_horizontal_lines": h_lines,
            "intersection_y_tolerance": 10,
        }
        p.to_image().debug_tablefinder(ts).save('Out.jpg')
        import pdb
        pdb.set_trace()
        
        # Get the bounding boxes of the tables on the page.
        bboxes = [table.bbox for table in p.find_tables(table_settings=ts)]
        table_texts, raw_texts = "", ""
        if len(bboxes) > 0:
            head = 0
            for idx, __bbox in enumerate(bboxes):
                x0, top, x1, bottom = __bbox
                table_texts += p.crop((0, head, p.width, top), relative=False, strict=False).extract_text()
                table_texts += f'<|page_{page_idx}_table_{idx+1}|>'
                head = bottom
            raw_texts = p.crop((0, head, p.width, p.height), relative=False, strict=False).extract_text()
        else:
            raw_texts = p.extract_text()
    
    if len(bboxes) > 0:
        bboxes = [(b[0] / p.width, b[1] / p.height, b[2] / p.width, b[3] / p.height)for b in bboxes]
        bboxes = sorted(bboxes, key=lambda x: x[1])
        
    return table_texts, raw_texts, bboxes


def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])


def get_embeddings(device_type="cuda"):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            query_instruction="Represent this sentence for searching relevant passages:",
        )

    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )


def split_contexts(context: str, chunk_size=1000, overlap=False):
    all_chunks = []
    chunk = []
    num_words = 0
    for sent in sent_tokenize(context.strip()):
        num_words += len(sent.split())
        chunk.append(sent)
        if num_words >= chunk_size:
            chunk_str = " ".join(chunk[:-1])
            all_chunks.append(chunk_str)

            if overlap:
                num_words -= len(chunk[0].split())
                chunk = chunk[1:]
            else:
                chunk = [chunk[-1]]
                num_words = len(chunk[0].split())
        
    if len(chunk) > 0:
        chunk_str = " ".join(chunk)
        all_chunks.append(chunk_str)
        chunk = []
        num_words = 0
        
    return all_chunks


def clean_text(text: str) -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords from text using regex
    stopwords_list = set(nltk.corpus.stopwords.words('english'))
    stopwords_pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    text = re.sub(stopwords_pattern, '', text)

    # Replace punctuation, newline, tab with space
    text = re.sub(r'[,.!?|]|[\n\t]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    text = text.split(" ")
    return text