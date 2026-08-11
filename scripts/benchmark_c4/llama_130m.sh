# LLaMA-130M, GaLore-Adam, 1 A100, 1 Node
BS=256
TOTAL_BS=512
TRAIN_STEPS=20000
WARMUP_STEPS=2000
LR=0.0025
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr ${LR} \
    --batch_size ${BS} \
    --total_batch_size ${TOTAL_BS} \
    --num_training_steps ${TRAIN_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 5000 \
    --optimizer adam \
    --eval_batch_size 512