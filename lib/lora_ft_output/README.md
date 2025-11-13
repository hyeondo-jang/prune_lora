---
library_name: transformers
base_model: /home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/Llama-2-7b-hf_wanda/0.6
tags:
- generated_from_trainer
datasets:
- c4
model-index:
- name: lora_ft_output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lora_ft_output

This model is a fine-tuned version of [/home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/Llama-2-7b-hf_wanda/0.6](https://huggingface.co//home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/Llama-2-7b-hf_wanda/0.6) on the c4 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.3226
- Perplexity: 10.2021

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 1
- seed: 0
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Perplexity |
|:-------------:|:-----:|:----:|:---------------:|:----------:|
| 2.7468        | 0.125 | 1    | 2.3641          | 10.6349    |
| 2.518         | 0.25  | 2    | 2.3561          | 10.5495    |
| 2.6993        | 0.375 | 3    | 2.3473          | 10.4577    |
| 2.6008        | 0.5   | 4    | 2.3402          | 10.3834    |
| 2.2852        | 0.625 | 5    | 2.3331          | 10.3100    |
| 2.5513        | 0.75  | 6    | 2.3284          | 10.2617    |
| 2.7164        | 0.875 | 7    | 2.3246          | 10.2221    |
| 2.6637        | 1.0   | 8    | 2.3226          | 10.2021    |


### Framework versions

- Transformers 4.45.0
- Pytorch 2.7.0+cu126
- Datasets 3.6.0
- Tokenizers 0.20.3
