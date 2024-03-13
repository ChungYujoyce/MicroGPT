FROM nvcr.io/nvidia/pytorch:23.08-py3
WORKDIR /workspace/
RUN apt-get update && \
 apt-get install -y poppler-utils
 
COPY ./requirements.txt .
RUN pip install --upgrade pip \
  && pip uninstall -y opencv \
  && pip install -r requirements.txt
  
RUN git clone --branch add-bm25 https://github.com/ChungYujoyce/langchain.git \
    && cd langchain/libs/langchain \
    && pip install -e .