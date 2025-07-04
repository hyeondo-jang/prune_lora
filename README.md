# Towards Extremely Sparse Large Language Models
Offical codebase for paper "Towards Extremely Sparse Large Language Models" (work in progress)


### 1. Setup

#### Venv setup
```bash
conda create -n gpa --python=3.10
conda activate gpa
pip install -r requirements.txt
```


### 2. Running Experiments

#### 
```bash
python main.py \
    --model="google/gemma-2-2b" \
    --prune_method="dense" \
    --sparsity_ratio=0.0 \
    --seqlen=2048 \
    --seed=0
```

```bash
python main.py \
    --model="google/gemma-2-2b" \
    --prune_method="wanda" \
    --sparsity_ratio=0.5 \
    --sparsity_type="unstructured" \
    --seqlen=2048 \
    --nsamples=128 \
    --dataset="c4" \
    --eval_zero_shot=True \
    --seed=0
```

We support [dense,sparsegpt,wanda,safe] for baseline experiment

#### GPA
```bash
python main.py \
    --model="google/gemma-2-2b" \
    --prune_method="global_admm" \
    --sparsity_ratio=0.5 \
    --sparsity_type="unstructured" \
    --seqlen=2048 \
    --admm_steps=4096 \
    --admm_batch_size=2 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lr=2e-4 \
    --admm_lmda=0.01 \
    --admm_interval=32 \
    --loss_type="ntp" \
    --eval_zero_shot=True \
    --seed=0
```


### 3. Multigpu (w.i.p)

```bash
torchrun --nproc_per_node=4 main.py \
    --model="meta-llama/Llama-2-7b-hf" \
    --prune_method="global_admm" \
    --sparsity_ratio=0.5 \
    --admm_epochs=1 \
    --admm_steps=100 \
    --seed=0
```


### 6. Configs

#### Model
- `--model`: Model to prune (예: "google/gemma-2-2b", "meta-llama/Llama-2-7b-hf")
- `--seqlen`: Context length (default: 2048)

#### Pruning
- `--prune_method`: Pruning method ("magnitude", "wanda", "sparsegpt", "safe", "alps", "global_admm", "dense")
- `--sparsity_ratio`: sparsity (0.0 ~ 1.0)
- `--sparsity_type`: sparsity type ("unstructured", "4:8", "2:4")

#### Dataset
- `--dataset`: calibration dataset ("c4", "wikitext2"), used only for local methods
- `--nsamples`: number of samples for calibration data

#### Evaluation
- `--eval_zero_shot`: Zero-shot evaluation






