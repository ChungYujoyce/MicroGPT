FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /workspace/
RUN apt-get update && \
 apt-get install -y poppler-utils
 
COPY ./requirements.txt .
RUN pip install --upgrade pip \
  && pip uninstall -y opencv \
  && pip uninstall -y apex \
  && pip install -r requirements.txt \
  && pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
  

RUN git clone --branch add-bm25-on-v0.1.2 https://github.com/ChungYujoyce/langchain.git \
    && cd langchain/libs/langchain \
    && pip install -e .
    