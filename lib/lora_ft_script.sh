CUDA_VISIBLE_DEVICES=0 python finetune_lm.py \
    --model_name_or_path /home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6 \
    --config_name "facebook/opt-125m" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 32786\
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir /home/hyeondojang/log_teams/opt-model/elsa_iclr26/lib/lora_ft/save

# CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
#     --model /home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6 \
#     --lora_weights /home/hyeondojang/log_teams/opt-model/elsa_iclr26/lib/lora_ft/save