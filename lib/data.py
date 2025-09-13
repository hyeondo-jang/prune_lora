from absl import logging
import os
import json
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer

# =====================================================================================
# Section 1: Unified Data Loading Logic (New Standard)
#
# This section contains the primary functions for data loading and processing.
# The main entry point is `get_dataset`, which returns a `datasets.Dataset` object.
# =====================================================================================

def _get_raw_dataset(dataset_name, data_type="train", data_path=None):
    """
    Loads the raw text data.
    If `data_path` is provided, it loads specific json.gz files from that local directory.
    Otherwise, it downloads from the Hugging Face Hub.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "c4" in dataset_name.lower():
        split_name = "train" if data_type == "train" else "validation"
        data_files_config = {
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz"
        }
        if data_path:
            if local_rank == 0:
                logging.info(f"Loading C4 raw data from local path: {data_path}")
            files_to_load = {split: os.path.join(data_path, fname) for split, fname in data_files_config.items()}
            return load_dataset(
                'json',
                data_files={split_name: files_to_load[split_name]},
                split=split_name,
            )
        else:
            if local_rank == 0:
                logging.info("Loading C4 raw data from Hugging Face Hub.")
            return load_dataset(
                'allenai/c4',
                data_files={split_name: data_files_config[split_name]},
                split=split_name,
                trust_remote_code=True,
                cache_dir='~/.cache/huggingface/datasets'
            )
            
    elif "wikitext2" in dataset_name.lower():
        # ... (wikitext2 및 ptb 로직은 변경 없음)
        split_name = "train" if data_type == "train" else "test"
        return load_dataset('wikitext', 'wikitext-2-raw-v1', split=split_name, trust_remote_code=True)
    elif "ptb" in dataset_name.lower():
        split_name = "train" if data_type == "train" else "validation"
        return load_dataset('ptb_text_only', 'penn_treebank', split=split_name, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def _process_and_tokenize(raw_dataset, dataset_name, tokenizer, nsamples, seqlen, seed):
    random.seed(seed)
    
    all_tokens = []
    if dataset_name.lower() == "c4":
        for _ in tqdm(range(nsamples), desc="Tokenizing C4"):
            while True:
                i = random.randint(0, len(raw_dataset) - 1)
                try:
                    text = raw_dataset[i]['text']
                    tokens = tokenizer(text, return_tensors='pt').input_ids
                    if tokens.shape[1] > seqlen:
                        all_tokens.append(tokens)
                        break
                except Exception:
                    continue # Skip samples that cause tokenization errors
    else: # wikitext2, ptb
        text_column = "text" if "wikitext" in dataset_name.lower() else "sentence"
        full_text = "\n\n".join(raw_dataset[text_column])
        all_tokens.append(tokenizer(full_text, return_tensors='pt').input_ids)

    processed_samples = []
    is_gemma_tokenizer = 'gemma' in tokenizer.__class__.__name__.lower()
    for _ in tqdm(range(nsamples), desc="Generating samples"):
        token_source = random.choice(all_tokens)
        
        slice_len = seqlen - 1 if is_gemma_tokenizer else seqlen
        
        start_index = random.randint(0, token_source.shape[1] - slice_len - 1)
        end_index = start_index + slice_len
        
        inp = token_source[:, start_index:end_index]

        if is_gemma_tokenizer:
            bos_tensor = torch.tensor([[tokenizer.bos_token_id]])
            inp = torch.cat([bos_tensor, inp], dim=1)
        
        processed_samples.append({
            "input_ids": inp.squeeze(0).tolist(),
            "attention_mask": [1] * seqlen,
            'labels': inp.squeeze(0).tolist()
        })
            
    return Dataset.from_list(processed_samples)

def get_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    nsamples: int,
    seed: int,
    seqlen: int,
    data_type: str = "train",
    cache_dir: str = "dataset",
    save_to_cache: bool = False,
    data_path: str = None
) -> Dataset:
    """
    Creates or loads a tokenized dataset.
    The `data_path` argument now controls where the RAW data is loaded from.
    The processed (tokenized) data is still cached locally for speed.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    safe_model_name = tokenizer.name_or_path.replace("/", "_")
    # data_path가 제공되었는지를 반영하는 캐시 디렉토리 이름 생성
    path_suffix = f"local_{Path(data_path).name}" if data_path else "hub"
    data_dir = Path(cache_dir) / safe_model_name / data_type / f"{dataset_name}_{nsamples}_{seqlen}_{path_suffix}"
    
    if data_dir.exists():
        if local_rank == 0:
            logging.info(f"Loading cached processed dataset from {data_dir}")
        return load_from_disk(str(data_dir))

    if local_rank == 0:
        logging.info(f"No valid processed cache found. Generating dataset...")
    raw_dataset = _get_raw_dataset(dataset_name, data_type, data_path=data_path)
    dataset = _process_and_tokenize(raw_dataset, dataset_name, tokenizer, nsamples, seqlen, seed)

    if save_to_cache:
        data_dir.mkdir(parents=True, exist_ok=True)
        if local_rank == 0:
            logging.info(f"Saving processed dataset to {data_dir}")
        dataset.save_to_disk(str(data_dir))
        
        metadata_file = data_dir / "metadata.json"
        metadata = {
            'tokenizer_name': tokenizer.name_or_path,
            'dataset_name': dataset_name,
            'data_type': data_type,
            'nsamples': nsamples,
            'seqlen': seqlen,
            'created_at': datetime.now().isoformat(),
            'source_data_path': data_path if data_path else 'Hugging Face Hub'
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    return dataset

# =====================================================================================
# Section 2: Backward Compatibility Layer for Local Solvers
#
# These functions provide a compatibility layer for older code that expects
# the `get_loaders` format. They now use the new `get_dataset` function internally.
# =====================================================================================

class TokenizerWrapper:
    """Wrapper for tokenized input IDs to maintain backward compatibility."""
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

def get_loaders(
    name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: AutoTokenizer = None,
    data_path: str = None
) -> tuple[list, TokenizerWrapper]:
    """
    Provides data in the legacy format for local pruning solvers.
    This function is now a wrapper around the new `get_dataset` function.

    Returns:
        tuple: (trainloader, testenc)
            - trainloader: A list of (input_tensor, target_tensor) tuples.
            - testenc: A TokenizerWrapper object containing test data.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logging.info("Using `get_loaders` (legacy compatibility mode).")

    # 1. Use the new, unified function to get the training dataset
    train_dataset = get_dataset(
        dataset_name=name,
        tokenizer=tokenizer,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        data_type="train",
        data_path=data_path
    )

    # 2. Convert the `datasets.Dataset` object to the old `trainloader` format
    trainloader = []
    for sample in train_dataset:
        inp = torch.tensor(sample['input_ids']).unsqueeze(0)  # (seqlen,) -> (1, seqlen)
        if 'gemma' in tokenizer.__class__.__name__.lower():
            inp[:, 0] = tokenizer.bos_token_id
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # 3. Create the test data object in the old format (using wikitext2 as default)
    if name.lower() == "c4":
        valdata = _get_raw_dataset("c4", "validation",data_path=data_path)
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors='pt')
        valenc = valenc.input_ids[:,:(256*seqlen)]
        valenc = TokenizerWrapper(valenc)
    elif name.lower() == "wikitext2":
        valdata = _get_raw_dataset("wikitext2", "validation",data_path=data_path)
        valenc = tokenizer("\n\n".join(valdata["text"]), return_tensors='pt')
        valenc = TokenizerWrapper(valenc.input_ids)

    return trainloader, valenc

# =====================================================================================
# Section 3: Utility Classes for Local Solvers (e.g., SAFE)
# =====================================================================================

class TensorData(torch.utils.data.Dataset):
    def __init__(self, data, targets, device):
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.targets)

class TensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False
        )