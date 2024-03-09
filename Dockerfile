FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /workspace/
RUN apt-get update && \
 apt-get install -y poppler-utils
 
RUN pip install --upgrade pip \
  && pip install langchain==0.0.267 \
  && pip install chromadb==0.4.6 \
  && pip install nltk \
  && pip install InstructorEmbedding \
  && pip install sentence-transformers==2.2.2 \
  && pip install faiss-cpu \
  && pip install huggingface_hub \
  && pip install transformers \
  && pip install rank-bm25 \
  && pip install auto_gptq \
  && pip install bitsandbytes \ 
  && pip install easyocr \
  && pip install pdf2image \
  && pip install accelerate \
  && pip install click \
  && pip install flask \
  && pip install pdfplumber \
  && pip install streamlit \
  && pip install Streamlit-extras \
  && pip uninstall -y opencv \
  && pip install opencv-python==4.8.0.74 \
  && pip install opencv-python-headless==4.8.0.74 \
  && pip install langchain-community
  
RUN git clone --branch add-bm25 https://github.com/ChungYujoyce/langchain.git \
    && cd langchain/libs/langchain \
    && pip install -e .