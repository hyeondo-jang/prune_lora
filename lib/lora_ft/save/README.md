---
library_name: peft
base_model: /home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6
tags:
- generated_from_trainer
datasets:
- c4
metrics:
- accuracy
model-index:
- name: save
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: c4
      type: c4
      split: None
    metrics:
    - type: accuracy
      value: 0.3246442965315095
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# save

This model is a fine-tuned version of [/home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6](https://huggingface.co//home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6) on the c4 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.9338
- Accuracy: 0.3246

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 5
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.45.0
- Pytorch 2.7.0+cu126
- Datasets 3.6.0
- Tokenizers 0.20.3