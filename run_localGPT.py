import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import VLLMOpenAI

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings, clean_text

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)


system_prompt = """As a helpful assistant, you will utilize the provided document to answer user questions. 
Read the given document before providing answers and think step by step. 
The document has an order of paragraphs with a higher correlation to the questions from the top to the bottom. 
The answer may be hidden in the tables, so please find it as closely as possible. 
Do not use any other information to answer the user. Provide a detailed answer to the question.
Also, please provide the answer in the following order of priorities if applicable:
Firstly, emphasize GPU characteristics and GPU products.
Secondly, Give prominence to power-related specifications such as fan cooling or liquid cooling, power consumption, and so on.
Thirdly, If applicable, mention green computing.
Remember, please don't provide any fabricated information, ensuring that everything stated is accurate and true."""

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    # if model_basename is not None:
    #     if ".gguf" in model_basename.lower():
    #         llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
    #         return llm
    #     elif ".ggml" in model_basename.lower():
    #         model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
    #     elif ".awq" in model_basename.lower():
    #         model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
    #     else:
    #         model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    # else:
    #     model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # # Load configuration from the model to avoid warnings
    # generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_tokens=MAX_NEW_TOKENS,
    #     # do_sample=False,
    #     temperature=0.0,
    #     # top_p=0.0,
    #     # top_k=1,
    #     # repetition_penalty=1.15,
    #     generation_config=generation_config,
    # )

    # local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    local_llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.7:5000/v1",
        model_name="test",
        max_tokens=512,
        temperature=0,
        model_kwargs={
            "stop": [],
        },
    )

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama3"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    
    # Return document size
    k = 4
    
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": k * 2})
    
    collections = db.get()
    documents = [Document(page_content=c, metadata=m) for m, c in zip(collections['metadatas'], collections['documents'])]
    retriever_bm25 = BM25Retriever.from_documents(documents=documents, preprocess_func=clean_text, k=k)
    
    # get the prompt template and memory if set by the user.
    # prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                            + system_prompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                            + """{question}<|eot_id|>""" + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # load the llm pipeline
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.7:5000/v1",
        model_name="test",
        max_tokens=512,
        temperature=0,
        model_kwargs={
            "stop": [],
        },
    )

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            retriever_bm25=retriever_bm25,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            retriever_bm25=retriever_bm25,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
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
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama", "llama3", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type='mistral' if 'mistralai' in MODEL_ID else model_type)
    qa = retrieval_qa_pipline(device_type, use_history)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
