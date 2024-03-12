## This project is inspired by [localGPT](https://github.com/PromtEngineer/localGPT) with multiple tailored enhancements for specific use cases.
> 1. Table-intense PDFs
> 2. Specific information retrieval with high accuracy

## Feature Highlights
1. Table parsing
2. RAG system (Knowledge database) management
3. VectorStore management
## Workflow Overview
![image](https://hackmd.io/_uploads/B1rlQwTa6.png)

## Environment Setup

1. Clone the repository
```
git clone https://github.com/ChungYujoyce/MicroGPT.git
```
2. Install [Docker](https://docs.docker.com/get-docker/) for virtual environment management.

*Note: Please make sure CUDA driver is compatible*

**Method 1:** Get the [Docker image](https://hub.docker.com/r/jj0122/microgpt/tags), start the container. (cuda version 12.2)
```
docker pull jj0122/microgpt:pt23.08
```
(The docker image is built from `Dockerfile` in this repository, you can modify if needed.)


**Method 2:** Get the [Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html) and start the container. (cuda version 12.3)
```
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```
Directly install dependencies in the docker using pip.
```
pip install -r requirements.txt
```

3. Optional:
Clone the repository to add my own modification on [langchain](https://github.com/ChungYujoyce/langchain/commit/3d2ade69c449fd34d0e80f7a7123a0c77495f1ff)
```
git clone --branch add-bm25 https://github.com/ChungYujoyce/langchain.git
cd langchain/libs/langchain
pip install -e .
```
This alteration caters to the requirement of users who wish to inspect specific details of a product. For further information, please refer to the 'Other Improvements' section for the details.
#### **Data Ingestion:**
Put your files in the `SOURCE_DOCUMENTS` folder. 
#### Support file formats:
Support PDF especially. Other types have not been supported yet.
#### Ingest 
```
python pipeline.py
```
#### Ask questions to your documents, locally!

Open [constants.py](https://github.com/ChungYujoyce/MicroGPT/blob/main/constants.py) in an editor of your choice and depending on choice add the LLM you want to use. By default, the following model will be used:
```
MODEL_ID='mistralai/Mixtral-8x7B-Instruct-v0.1'
MODEL_BASENAME=None
```
In order to chat with your documents, run the following command (by default, it will run on `cuda`).
```
python run_localGPT.py
```
You can use the `-s` flag with run_localGPT.py to show which chunks were retrieved by the embedding model. By default, it will show 4 different sources/chunks. You can change the number of sources/chunks
```
python run_localGPT.py -s
```
**Run the GUI**

1. Run `python run_localGPT_API.py`
2. Navigate to the `/LOCALGPT/localGPTUI` directory.
3. Run `python localGPTUI.py`
4. Open up a web browser and go the address `http://localhost:3111/`

## Technical details
## Data Pre-processing (Fine-tune on PDFs)

**LocalGPT:**
* `ingest.py`: 
    * Step 1: Use `PDFMinerLoader` to read PDFs.
    * Step 2: Split the texts into chunks with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` and send to Chroma DB directly.


**Downsides:**
1. When encountering tables, it parses the table content row-by-row, which fails to capture the relationships between each cell and loses the meaning of the table.
2. It is not possible to see the processed results of the input materials, which makes it difficult to improve data quality down the road.
3. It fully depends on a third-party parsing tool and is difficult to fine-tune.
4. The table layout on the frontend is not user-friendly.
5. If upload files from UI interface, it re-run the `ingest.py` all over again.

**MicroGPT:**
* `pdf_prep.py`: 
    * Step 1: PDF to Image transformation for the model to process.
    * Step 2: Table parsing (cell/column/row) with TableTransformer and OCR model.
        * Table Transformer can detect tables on each page and crop them out. 
        * The OCR model can identify the characters in the table.
    * Step 3: Save as `.csv` file as it's a better data structure than pure lists.
    * Step 4: Extract texts EXCLUDING the tables.
    * Step 5: Concatenate the tables and texts together within each page and save as `.txt`.

* `chunk_prep.py`:
    * Filter out useless sentences using regex (e.g. `©2024 Copyright Super Micro Computer, Inc. All rights reserved.`)
    * Implemented my own `split_contexts(chunk_size=300, overlap=False)` (sentence-based splitting) to split texts into chunks.
        * Note : `RecursiveCharacterTextSplitter` has chances to cut sentense in half.
    * Transform `.txt` to Langchain compatible `Document` type. 

* `pipeline.py`:
    * Link all components together and pass to Chroma DB ingestion.

*Note: Choosing OCR is a trade-off between the table structure and word misspelling.*

localGPT Result (No imtermediate records and management, raw output is as the following):
![Screenshot 2024-03-04 at 10.41.49 AM](https://hackmd.io/_uploads/BJCl85mTp.png)

## Knowledge Database Management

```
/PARSED_DOCUMENTS
xx1.pdf                          # Each PDF has its own folder
    /intermediate
        /page_1                  # Each page has its own folder
            /page.png            
            /raw_text.txt        # 
            /table_1.csv         # table content (OCR)
            /table_1.png         # table image cropped by table-transformer
            /table_text.txt      # 
        /page_2
        ...
    /paragraphs                  # folder to save all chunks
        /chunk_1.txt
        /chunk_2.txt
        ...
xx2.pdf
...
```

Result: (easier to revise and manage)
table_1.csv
![Screenshot 2024-03-04 at 12.08.41 PM](https://hackmd.io/_uploads/SkLI5smpT.png)

(chunk_1.txt, human-friendly layout):
```
DATASHEET
GPU SuperServer SYS-421GU-TNXR
Universal 4U Dual Processor (4th and 5th Gen Intel Scalable Processors) GPU System with NVIDIA HGX™ H100 4-GPU
SXM5 board, NVLINK™ GPU-GPU Interconnect, and Redundant 3000W Titanium Level Power Supplies. More details here
Key Applications
High Performance Computing, AI/Deep Learning Training, Large Language
Model (LLM) Natural Language Processing,
Key Features
32 DIMM slots Up to 8TB: 32x 256 GB DRAM Memory Type: 4800MHz ECC
DDR5; 32 DIMM slots Up to 8TB: 32x 256 GB DRAM Memory Type: 5600MTs
ECC DDR5;
8 PCIe Gen 5.0 X16 LP Slots;
; Flexible networking options;
2 M.2 NVMe and SATA for boot drive only; 6x 2.5" Hot-swap NVMe/SATA/SAS
drive bays;

Form Factor | 4U Rackmount
,"Enclosure: 449x 175.6 X 833mm (17.67"" x 7"" x32.79"") Package: 700 x 370 x 1260mm (27.55"" x 14.57"" x49.6"")"
Processor | Dual Socket E (LGA-4677) Sth Gen Intele Xeone/4th Gen Intele Xeone Scalable processors Up to 56C/112T; Up to 112.5MB Cache per CPU
GPU | Max GPU Count: Up to 4 onboard GPU(s) Supported GPU: NVIDIA SXM: HGX H1OO 4-GPU (8OGB) CPU-GPU Interconnect: PCle 5.0 xl6 CPU-to-GPU Interconnect GPU-GPU Interconnect: NVIDIA? NVLinke
System Memory | Slot Count: 32 DIMM slots Max Memory (2DPC): Up to 8TB 5600MT/s ECC DDRS
Drive Bays | "6x 2.5"" hot-swap NVMe/SATA drive bays (6x 2.5"" NVMe hybrid)"
Expansion Slots | 1 PCle 5.0 xl6 LP slot(s) 7 PCle 5.0 X16 slot(s)
On-Board Devices | Chipset: Intele C741 Network Connectivity: 2x 1OGbE BaseT with Intele X710-AT2 IPMI: Support for Intelligent Platform Management Interface V.2.0 IPMI 2.0 with virtual media over LAN and KVM-over-LAN support
Input Output | Video: 1 VGA port(s)
```

## VectorStore Management

* db_management.py
    * Supports `UPDATE`, `DELETE` functions.
    * `mapping.json`: Create a ID and Source mapping library for user (admin) to clearly view the status in a timely manner.
* db_mng.py
    * Class version of `db_management.py` for convenient function calls.
* run_localGPT_API.py
    * Reconstruct the logic of adding file so that when ingesting the data, all sources already existing in the knowledge database won't be reprocessed.

## Frontend UI Supports
* User are allowed to `update`/ `delete` the chunks that display as the sources of the answer, by doing this, users can directly make contribution of improving the database.
* Enhance the table layout to improve user-friendliness, facilitating easier navigation and information retrieval for users.
* Add a "Cancel" button for users to discard the changes.

## Other Improvements

### 1. Retriever:
Add [BM25](https://python.langchain.com/docs/integrations/retrievers/bm25) in langchain to get two scores:
* Similarity score: from vectorestore retrieval techniques.
* **[ADD] Critical terms hitting score:**
Expanded the number of returned documents to 8 to ensure comprehensive coverage and minimize the risk of missing relevant information. (But the result outputs still remains 4) Subsequently, the cleaned user-query and documents are passed to `bm25_retriever.get_relevant_documents`. 
> Original query: What's the memory and GPU specifications of SRS-42UGPU-AI-SU2?
> Cleaned query: ['memory', 'gpu', 'specifications', 'srs-42ugpu-ai-su2']


When the query aligns with terms present in a document, that document receives a higher score using `bm25_retriever`. *(I used "-" to distinguish it as our product names usually have hyphens)* This approach proves particularly beneficial when seeking specific information (e.g. product name) rather than broad concepts, complementing the strengths of the original retriever.
### 2. Mixtral model with prompt template fine-tuning:
* Instruct the model to prioritize table information.
* Remove potentially error-prone leading and trailing white spaces from the promt template.
* Add `do_sample=False` to ensure that each output token is selected based on the highest probability, rather than randomly sampling from the output probability distribution.
* Remove `repetition_penalty` to ensure the highest level of result accuracy.

### 3. Table format adjustment:
Switching the model to Mixtral did not significantly improve the performance as I expected, especially when querying table-related questions. Upon investigation, I discovered that the CSV table format, with comma separation, posed challenges for the model in recognizing table structures. Consequently, I opted for a different format, resulting in a remarkable improvement in the results.
```
Column 1,Column 2,Column 3            Column 1 | Column 2 | Column 3
item 1,item 3,item 5           -->    item 1 | item 3 | item 5 
item 2,item 4,item 6                  item 2 | item 4 | item 6
```

*Note: the table format of is worse.*
```
<table border="1">
    <tr>
        <th>Column 1</th>
        <th>Column 2</th>
    </tr>
    <tr>
        <td>Row 1, Cell 1</td>
        <td>Row 1, Cell 2</td>
    </tr>
    <tr>
        <td>Row 2, Cell 1</td>
        <td>Row 2, Cell 2</td>
    </tr>
</table>
``` 
    

### 4. Table detection improvement:
At times, the table transformer may encounter difficulty detecting tables within certain PDFs, occurring approximately 10% of the time in my experiments. As an alternative approach, I utilized the built-in pdfplumber table bounding box detection to locate the table's position and pass it (as an object) to the OCR model.

 
## To-Do [In the order of Priority]
1. Speed of Mixtral
2. OCR model improvement
https://github.com/dvlab-research/LongLoRA/tree/main/pdf2txt













