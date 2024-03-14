CUDA_VISIBLE_DEVICES=0,1 python vllm_server.py \
    --model=/workspace/models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83 \
    --max-model-len=128 \
    --tensor-parallel-size=2 \
    --pipeline-parallel-size=1 \
    --trust-remote-code \
    --dtype bfloat16 \