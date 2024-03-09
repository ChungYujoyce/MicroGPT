## This project is based on [localGPT](https://github.com/PromtEngineer/localGPT) with multiple tailored enhancements for specific use cases.
## User Guide
[UPDATE] Get new Docker image to use CUDA 12.2 and replace conda environment.
```
nvcr.io/nvidia/pytorch:24.01-py3
```
Directly install dependencies in the docker.
```
pip install -r requirements.txt
```
Installing LLAMA-CPP: For `NVIDIA` GPUs support, use `cuBLAS`

If ran into error, run this
```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```
Run `python pipeline.py` to ingest your data.

## Workflow Overview
![Untitled](https://hackmd.io/_uploads/Sy9NVn766.jpg)
## Data Pre-processing (Fine-tune on PDFs first)
**LocalGPT:**
* `ingest.py`: 
    * Step 1: Use `PDFMinerLoader` to read PDFs.
    * Step 2: Split the texts into chunks with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` and send to Chroma DB directly.
               


**Downsides:**
1. When encountering tables, it parses the table content row-by-row, which fails to capture the relationships between each cell and loses the meaning of the table.
2. It is not possible to see the processed results of the input materials, which makes it difficult to improve data quality down the road.
3. It fully depends on a third-party parsing tool and is difficult to fine-tune.

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

## Knowledge Database

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

Form Factor,4U Rackmount
,"Enclosure: 449x 175.6 X 833mm (17.67"" x 7"" x32.79"") Package: 700 x 370 x 1260mm (27.55"" x 14.57"" x49.6"")"
Processor,Dual Socket E (LGA-4677) Sth Gen Intele Xeone/4th Gen Intele Xeone Scalable processors Up to 56C/112T; Up to 112.5MB Cache per CPU
GPU,Max GPU Count: Up to 4 onboard GPU(s) Supported GPU: NVIDIA SXM: HGX H1OO 4-GPU (8OGB) CPU-GPU Interconnect: PCle 5.0 xl6 CPU-to-GPU Interconnect GPU-GPU Interconnect: NVIDIA? NVLinke
System Memory,Slot Count: 32 DIMM slots Max Memory (2DPC): Up to 8TB 5600MT/s ECC DDRS
Drive Bays,"6x 2.5"" hot-swap NVMe/SATA drive bays (6x 2.5"" NVMe hybrid)"
Expansion Slots,1 PCle 5.0 xl6 LP slot(s) 7 PCle 5.0 X16 slot(s)
On-Board Devices,Chipset: Intele C741 Network Connectivity: 2x 1OGbE BaseT with Intele X710-AT2 IPMI: Support for Intelligent Platform Management Interface V.2.0 IPMI 2.0 with virtual media over LAN and KVM-over-LAN support
Input Output,Video: 1 VGA port(s)
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
* User are allowed to update/delete the chunks that display as the sources of the answer. 

## To-Do [In the order of Priority]
1. Retriever is bad:
Need to modify Langchain source code to change (sort/filter) the output sources. 
Build a rule-based retriever to tailor our needs.
2. Model is bad:
Tried both llama 7B and 70B model, still bad.
Potentially change to OpenAI or Mixtral model. [TBD]
3. OCR model improvement
https://github.com/dvlab-research/LongLoRA/tree/main/pdf2txt






