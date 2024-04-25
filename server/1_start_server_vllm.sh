HF_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
HF_CHECKPOINT_PATH="/data/nlp/${HF_MODEL}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server  \
    --model=${HF_CHECKPOINT_PATH} \
    --served-model-name='llama3-8b-instruct' \
    --max-model-len=4096 \
    --tensor-parallel-size=1 \
    --pipeline-parallel-size=1 \
    --trust-remote-code \
    --dtype bfloat16 \