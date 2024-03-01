import os, re
from langchain.docstore.document import Document
from utils import split_contexts


def text_filter(text):
    # take off copyright line
    idx = re.search(r"©", text).span()[0]
    return text[:idx]


def text_to_chunk(table_dict, text_dict, dis_dir) -> Document:

    chunks =  []
    pages = len(text_dict)

    for page_idx in range(pages):
        text = ""
        table_text, raw_text = text_dict[page_idx]['table_text'], text_dict[page_idx]['raw_text']
        if table_text:
            table_list = table_dict[page_idx]
            for table in table_list:
                table_name = table.split('.')[0].split('/')[-1]
                if table_text.find(table_name) != -1:
                    table_content = open(table, "r").read()
                    table_text = table_text.replace(f"<|page_{page_idx+1}_"+table_name+"|>", "\n\n"+table_content+"\n\n")
            chunks.append(table_text)
            table_dict[page_idx] = []

        text += raw_text
        clean_text = text_filter(text)
        chunks += split_contexts(clean_text, chunk_size=300, overlap=False)

    # Tables cannot be recognized by pdfplumber
    chunks += [f"\n\n{open(table, 'r').read()}" for page, tables in table_dict.items() for table in tables if tables != []]

    final_chunks = []
    curr_text = ""

    for chunk in chunks:
        if len(split_contexts(curr_text + chunk, chunk_size=300, overlap=False)) == 1:
            curr_text += chunk
        else:
            final_chunks.append(curr_text)
            curr_text = chunk

    final_chunks.append(curr_text)

    ## [TODO]
    ## 1. shrink chunk size if possible
    docs = []
    for i in range(len(final_chunks)):
        with open(f'{dis_dir}/chunk_{i+1}.txt', 'w', encoding='utf-8') as f:
            f.write(final_chunks[i])
        
        # transform txt chunks into langchain Document type
        doc = Document(page_content=final_chunks[i], metadata={"source": f'{dis_dir}/chunk_{i+1}.txt'})
        docs.append(doc)

    return docs



