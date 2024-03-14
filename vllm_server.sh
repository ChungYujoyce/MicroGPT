CUDA_VISIBLE_DEVICES=0,1 python vllm_server.py \
    --model=/data/nlp/Mixtral-8x7B-Instruct-v0.1 \
    --max-model-len=128 \
    --tensor-parallel-size=2 \
    --pipeline-parallel-size=1 \
    --trust-remote-code \
    --dtype bfloat16 \