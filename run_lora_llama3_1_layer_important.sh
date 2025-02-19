export MODELSCOPE_CACHE=/mnt2/cache
export HF_DATASETS_CACHE=/mnt2/cache
MASTER_PORT=29508 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /mnt2/name/model/llama3_1_8b_instruct \
    --train_type lora \
    --model_type llama3_1 \
    --dataset mmlu.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --save_strategy steps \
    --save_steps 40 \
    --eval_strategy no \
    --model_author swift \
    --model_name swift-robot \
    --logging_steps 5 \
    --torch_dtype bfloat16 \
    --save_total_limit 20 \
    --output_dir /mnt2/name_prune/saves/llama3-1-8b-instruct/lora/mmlu \
    --gradient_checkpointing true \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 32
