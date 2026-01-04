docker run -d \
    --restart=always \
    --name qwen3_14b_base \
    --gpus=all \
    -e CUDA_VISIBLE_DEVICES="7" \
    --shm-size=10gb \
    --ipc=host \
    -v /data01/LLM_model/Qwen3-14B/:/model \
    -p 6093:20020 \
    vllm/vllm-openai:latest \
    --host 0.0.0.0 \
    --model /model \
    --tensor-parallel-size 1 \
    --load-format safetensors \
    --port 20020 \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen3_32b