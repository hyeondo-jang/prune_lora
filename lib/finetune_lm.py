#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List 

import datasets
import evaluate
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from absl import logging as absl_logging, app, flags
from data import get_dataset
from eval import eval_ppl, eval_zero_shot
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# 수정
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
try:
    # Newer PEFT versions (>=0.9) expose prepare_model_for_kbit_training
    from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training
except ImportError:
    try:
        # Older PEFT versions still export prepare_model_for_int8_training at top level
        from peft import prepare_model_for_int8_training
    except ImportError:
        try:
            # Fallback for intermediate versions
            from peft.tuners.lora import prepare_model_for_int8_training
        except ImportError:
            from peft.utils.peft_utils import prepare_model_for_int8_training
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.29.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."},
    )
    model_type: Optional[str] = field(default=None, metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "parameter lora_r"},
    )
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "parameter lora_alpha"}
    ,)
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "parameter lora_dropout"},
    )
    # lora_target_modules: Optional[List[str]] = field(
    #     default=["q_proj","v_proj"],
    #     metadata={"help": "parameter lora_target_modules"},
    # )
    config_overrides: Optional[str] = field(default=None, metadata={"help": "Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(default=False, metadata={"help": "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models)."},
    )
    torch_dtype: Optional[str] = field(default=None, metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights. choices: auto, bfloat16, float16, float32"},
    )
    low_cpu_mem_usage: bool = field(default=False, metadata={"help": "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. set True will benefit LLM loading time and RAM consumption."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."},
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(default=None, metadata={"help": "Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def main(argv):
    # Use absl flags if model_name_or_path is provided, otherwise fall back to HfArgumentParser
    global FLAGS
    
    # Get local_rank for distributed training (similar to main.py)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Check if flags are defined and model_name_or_path is provided
    use_flags = (hasattr(FLAGS, 'model_name_or_path') and FLAGS.model_name_or_path is not None)
    
    # Initialize wandb if enabled (only on main process, like main.py)
    if has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb and local_rank in [-1, 0]:
        wandb_project = FLAGS.wandb_project if hasattr(FLAGS, 'wandb_project') else 'lora-finetune'
        wandb.init(project=wandb_project)
        if hasattr(FLAGS, 'wandb_run_name') and FLAGS.wandb_run_name:
            wandb.run.name = FLAGS.wandb_run_name
        
        # Log all flags to wandb config (like main.py, update all arguments at once)
        if use_flags:
            # Get all flag values as dict (like main.py)
            arguments = FLAGS.flag_values_dict()
            if not dict(wandb.config):  # Only update if config is empty
                wandb.config.update(arguments)
            else:
                # If config already exists (e.g., from sweep), update with current values
                updated_args = {
                    k: wandb.config.get(k, v) for k, v in arguments.items()
                }
                wandb.config.update(updated_args, allow_val_change=True)
    
    if use_flags:
        # Use absl flags
        model_name_or_path = FLAGS.model_name_or_path
        config_name = FLAGS.config_name
        dataset_name = FLAGS.dataset_name
        output_dir = FLAGS.output_dir
        per_device_train_batch_size = FLAGS.per_device_train_batch_size
        per_device_eval_batch_size = FLAGS.per_device_eval_batch_size
        learning_rate = FLAGS.learning_rate
        num_train_epochs = FLAGS.num_train_epochs
        steps = FLAGS.steps
        eval_samples = FLAGS.eval_samples
        block_size = FLAGS.block_size
        do_train = FLAGS.do_train
        do_eval = FLAGS.do_eval
        overwrite_output_dir = FLAGS.overwrite_output_dir
        seed = FLAGS.seed
        lora_r = FLAGS.lora_r
        lora_alpha = FLAGS.lora_alpha
        lora_dropout = FLAGS.lora_dropout
        batch_size = FLAGS.batch_size
        lr_scheduler_type = FLAGS.lr_scheduler_type
        logging_steps = FLAGS.logging_steps
        eval_steps = FLAGS.eval_steps
        gradient_checkpointing = getattr(FLAGS, 'gradient_checkpointing', False)
        mixed_precision = FLAGS.mixed_precision
        device_map = FLAGS.device_map if FLAGS.device_map != 'None' else None
        use_torch_compile = FLAGS.use_torch_compile
        
        # Calculate gradient_accumulation_steps from batch_size
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if batch_size > 0:
            # Ensure batch_size is divisible by (per_device_train_batch_size * num_gpus)
            effective_per_device = per_device_train_batch_size * num_gpus
            if batch_size % effective_per_device != 0:
                logger.warning(
                    f"batch_size ({batch_size}) is not divisible by "
                    f"per_device_train_batch_size * num_gpus ({effective_per_device}). "
                    f"Using gradient_accumulation_steps = {batch_size // effective_per_device} "
                    f"(effective batch size will be {batch_size // effective_per_device * effective_per_device})"
                )
            gradient_accumulation_steps = max(1, batch_size // effective_per_device)
            actual_batch_size = gradient_accumulation_steps * effective_per_device
            if actual_batch_size != batch_size:
                logger.warning(
                    f"Requested batch_size={batch_size}, but actual effective batch_size={actual_batch_size} "
                    f"(per_device={per_device_train_batch_size}, gradient_accumulation={gradient_accumulation_steps}, num_gpus={num_gpus})"
                )
        else:
            gradient_accumulation_steps = 1
            actual_batch_size = per_device_train_batch_size * num_gpus
        
        # Calculate train_samples: actual_batch_size * steps
        # Each step uses actual_batch_size samples, so total samples = actual_batch_size * steps
        train_samples = actual_batch_size * steps
        
        # Create TrainingArguments from flags
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            do_train=do_train,
            do_eval=do_eval,
            overwrite_output_dir=overwrite_output_dir,
            seed=seed,
            fp16=(mixed_precision == 'fp16'),  # FP16 mixed precision
            bf16=(mixed_precision == 'bf16'),  # BF16 mixed precision
            logging_steps=logging_steps,
            optim="adamw_torch",  # AdamW optimizer
            save_strategy="no",  # Disable checkpoint saving
            evaluation_strategy="steps" if do_eval else "no",  # Evaluate every eval_steps
            eval_steps=eval_steps,
            group_by_length=False,
            report_to=[],  # We handle wandb logging manually (like main.py), so disable Trainer's automatic wandb
            lr_scheduler_type=lr_scheduler_type,  # Learning rate scheduler type
            warmup_steps=0,  # No warmup
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,  # Enable/disable gradient checkpointing
        )
            
        # Create ModelArguments and DataTrainingArguments from flags
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            config_name=config_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        data_args = DataTrainingArguments(
            dataset_name=dataset_name,
            train_samples=train_samples,  # Calculated from batch_size * steps
            eval_samples=eval_samples,
            block_size=block_size,
        )
        
        # Update wandb config with calculated train_samples and training settings
        if has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb and local_rank in [-1, 0]:
            wandb.config.update({
                'train_samples': train_samples,
                'requested_batch_size': batch_size,
                'actual_effective_batch_size': actual_batch_size,
                'per_device_train_batch_size': per_device_train_batch_size,
                'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
                'num_gpus': num_gpus,
                'warmup_steps': training_args.warmup_steps,
            }, allow_val_change=True)  # Allow value changes for calculated values
    else:
        # Fall back to HfArgumentParser (for backward compatibility)
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {getattr(training_args, 'bf16', False) or getattr(training_args, 'fp16', False)}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected at {last_checkpoint}, but skipping resume due to PyTorch 2.6 weights_only issue. "
                "Set --overwrite_output_dir to start fresh or manually delete the checkpoint."
            )
            last_checkpoint = None  # Skip checkpoint resume to avoid PyTorch 2.6 weights_only error

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # num_train_samples = retrain_args.max_steps * retrain_args.per_device_train_batch_size * retrain_args.gradient_accumulation_steps * world_size

    
    # raw_datasets = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'}
    # )

    # if "validation" not in raw_datasets.keys():
    #     raw_datasets["validation"] = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         split=f"train[:{data_args.validation_split_percentage}%]",
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         streaming=data_args.streaming,
    #     )
    #     raw_datasets["train"] = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         split=f"train[{data_args.validation_split_percentage}%:]",
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         streaming=data_args.streaming,
    #     )
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.config_name, use_fast=False)

    ## we use the tokenizer from vicuna
    if "decapoda-research" in model_args.config_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-13b-delta-v0",
            cache_dir=model_args.cache_dir,
            padding_side="right",
            use_fast=True,
        )

    # Get model loading options from flags (if using flags)
    if use_flags:
        device_map_value = device_map if device_map != 'None' else "auto"
        use_compile = use_torch_compile
        mixed_prec = mixed_precision
    else:
        device_map_value = "auto"  # Default for backward compatibility
        use_compile = False
        mixed_prec = "bf16"
    
    # Load model in FP16 first (for memory efficiency during loading)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.float16, 
        cache_dir=model_args.cache_dir, 
        low_cpu_mem_usage=True, 
        device_map=device_map_value
    )

    ############################################################################################
    # Upcast to FP32 if using BF16 mixed precision (same as retrain.py)
    # BF16 mixed precision requires FP32 model parameters, only forward/backward uses BF16
    if mixed_prec == 'bf16' and model.dtype == torch.float16:
        model = model.to(torch.float32)
        torch.cuda.empty_cache()  # Clear cache after dtype conversion
        logger.info("Model upcasted to FP32 for BF16 mixed precision training")
    
    # Prepare model for LoRA training (same as Wanda code)
    # Note: prepare_model_for_int8_training doesn't actually quantize to int8,
    # it just prepares the model for LoRA fine-tuning (freezes embeddings, etc.)
    model = prepare_model_for_int8_training(model)
    
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj","v_proj"],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    ############################################################################################
    
    train_dataset = get_dataset(
        "c4",
        tokenizer,
        nsamples=data_args.train_samples,
        seed=0,
        seqlen=2048,
        data_path=None
    )
    
    eval_dataset = get_dataset(
        "c4",
        tokenizer,
        nsamples=data_args.eval_samples,
        data_type="validation",
        seed=0,
        seqlen=2048,
        data_path=None
    )
    
    # Move model to GPU if device_map was None (model is on CPU)
    # Note: Only needed if device_map was None
    if device_map_value is None and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        torch.cuda.empty_cache()  # Clear cache after moving model
    
    ############################################################################################

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = list(raw_datasets["train"].features)
    # else:
    #     column_names = list(raw_datasets["validation"].features)
    # text_column_name = "text" if "text" in column_names else column_names[0]

    # # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # def tokenize_function(examples):
    #     with CaptureLogger(tok_logger) as cl:
    #         output = tokenizer(examples[text_column_name])
    #     # clm input could be much much longer than block_size
    #     if "Token indices sequence length is longer than the" in cl.out:
    #         tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #             " before being passed to the model."
    #         )
    #     return output

    # with training_args.main_process_first(desc="dataset map tokenization"):
    #     if not data_args.streaming:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on dataset",
    #         )
    #     else:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             remove_columns=column_names,
    #         )

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
    #             " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
    #             " override this default with `--block_size xxx`."
    #         )
    #         block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with training_args.main_process_first(desc="grouping texts together"):
    #     if not data_args.streaming:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc=f"Grouping texts in chunks of {block_size}",
    #         )
    #     else:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #         )

    # if training_args.do_train:
    #     if "train" not in tokenized_datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = lm_datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    #         train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        # if "validation" not in tokenized_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        # eval_dataset = lm_datasets["validation"]
        # if data_args.max_eval_samples is not None:
        #     max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        #     eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits  # Return logits for perplexity calculation

        def compute_metrics(eval_preds):
            # Calculate perplexity from logits and labels
            logits, labels = eval_preds
            
            # Convert to torch tensors if numpy arrays
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate loss
            loss = loss_fct(shift_logits, shift_labels)
            
            # Calculate perplexity
            try:
                perplexity = math.exp(loss.item())
            except OverflowError:
                perplexity = float("inf")
            
            return {"perplexity": perplexity}

    ################################################################################################################
    # Wandb logging callback (to log training metrics manually)
    class WandbLoggingCallback(TrainerCallback):
        def __init__(self):
            self.enabled = has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb and local_rank in [-1, 0]
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.enabled and logs is not None:
                wandb.log(logs, step=state.global_step)
    
    wandb_callback = WandbLoggingCallback() if (has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb) else None
    ################################################################################################################
    # Override training args if not using flags (backward compatibility)
    use_flags = (hasattr(FLAGS, 'model_name_or_path') and FLAGS.model_name_or_path is not None)
    if not use_flags:
        batch_size = 32
        training_args.gradient_accumulation_steps = batch_size // training_args.per_device_train_batch_size
        training_args.warmup_steps = 0  # No warmup
        training_args.lr_scheduler_type = "linear"  # Linear learning rate scheduling
        # Mixed precision is set from Args if using HfArgumentParser
        if not hasattr(training_args, 'fp16') and not hasattr(training_args, 'bf16'):
            training_args.bf16 = True  # Default to BF16 for backward compatibility
        training_args.logging_steps = 10  # Default value if not using flags
        training_args.optim = "adamw_torch"
        training_args.save_strategy = "no"  # Disable checkpoint saving
        training_args.evaluation_strategy = "steps" if training_args.do_eval else "no"  # Evaluate every eval_steps
        # eval_steps is already set from Args (HfArgumentParser), use that value
        if not hasattr(training_args, 'eval_steps') or training_args.eval_steps is None:
            training_args.eval_steps = 10  # Default value only if not set
        training_args.group_by_length = False
        training_args.report_to = []  # We handle wandb logging manually, so disable Trainer's automatic wandb
    else:
        # Ensure evaluation_strategy is set when using flags
        if not hasattr(training_args, 'evaluation_strategy') or training_args.evaluation_strategy is None:
            training_args.evaluation_strategy = "steps" if do_eval else "no"
    ################################################################################################################

    # Initialize our Trainer
    callbacks = []
    if wandb_callback is not None:
        callbacks.append(wandb_callback)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=callbacks if callbacks else None,
    )

    ############## code imported from alpaca-lora ###################
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # Use torch.compile if enabled (same as Wanda code)
    if use_compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")
    ############## code imported from alpaca-lora ###################

    # Training
    if training_args.do_train:
        checkpoint = None
        # Skip checkpoint resume to avoid PyTorch 2.6 weights_only error with numpy objects
        # If you need to resume, manually set resume_from_checkpoint or use older PyTorch version
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # Disable automatic checkpoint resume due to PyTorch 2.6 compatibility issue
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # Model saving disabled - user doesn't need to save the model
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        # model.save_pretrained(training_args.output_dir)
        # torch.save(trainer.model.state_dict(), f"{training_args.output_dir}/adapter_model.bin")

        metrics = train_result.metrics

        train_samples_count = (
            data_args.train_samples if data_args.train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(train_samples_count, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        eval_samples = data_args.eval_samples if data_args.eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    # Post-training evaluation: PPL and Zero-shot
    # Only run on main process (rank 0)
    if training_args.local_rank in [-1, 0]:
        logger.info("=" * 60)
        logger.info("Starting post-training evaluation (PPL and Zero-shot)")
        logger.info("=" * 60)
        
        # Get the trained model (with LoRA adapters)
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module  # Unwrap DDP/FSDP wrapper if present
        
        # Cast model to FP16 for evaluation (same as main.py)
        # This reduces memory usage during evaluation
        if "gemma-2-27b" in model_name_or_path:
            logger.info("gemma-2-27b model detected. Casting to torch.bfloat16 for stability.")
            model = model.to(torch.bfloat16)
        else:
            logger.info(f"Casting model to torch.float16 for evaluation.")
            model = model.to(torch.float16)
        
        # Get device
        device = training_args.device if hasattr(training_args, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Set seqlen attribute for eval_ppl (required by eval_ppl)
        if not hasattr(model, 'seqlen'):
            # Try to get seqlen from data_args or use default
            seqlen = getattr(data_args, 'block_size', 2048) if hasattr(data_args, 'block_size') else 2048
            model.seqlen = seqlen
            logger.info(f"Set model.seqlen = {seqlen}")
        
        # Prepare args-like object for eval functions
        class EvalArgs:
            def __init__(self):
                self.seed = training_args.seed
                self.data_path = getattr(FLAGS, 'data_path', None) if use_flags else None
        
        eval_args = EvalArgs()
        
        # PPL evaluation
        logger.info("--- Evaluating Perplexity (PPL) ---")
        try:
            ppl_test = eval_ppl(eval_args, model, tokenizer, device, data_path=eval_args.data_path)
            logger.info(f"PPL results: {[(key, ppl) for key, ppl in ppl_test.items()]}")
            
            if has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb:
                wandb.log({f"ppl_test({key})": value for key, value in ppl_test.items()})
        except Exception as e:
            logger.error(f"Error during PPL evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Zero-shot evaluation (optional)
        eval_zero_shot_flag = getattr(FLAGS, 'eval_zero_shot', False) if use_flags else False
        if eval_zero_shot_flag:
            logger.info("--- Evaluating Zero-shot Tasks ---")
            try:
                # Determine if accelerate is needed (for large models)
                accelerate = "70b" in model_name_or_path or "65b" in model_name_or_path
                task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "race"]
                num_shot = 0
                
                results_after = eval_zero_shot(eval_args, model_name_or_path, model, tokenizer, task_list, num_shot, accelerate)
                logger.info(f"Zero-shot results after LoRA fine-tuning:")
                logger.info(results_after)
                
                if has_wandb and hasattr(FLAGS, 'wandb') and FLAGS.wandb:
                    for task_name, metrics in results_after.items():
                        try:
                            acc = metrics.get('acc,none', metrics.get('acc', None))
                            stderr = metrics.get('acc_stderr,none', metrics.get('acc_stderr', None))
                            if acc is not None:
                                wandb.log({f"lora_ft/{task_name}_acc": acc})
                            if stderr is not None:
                                wandb.log({f"lora_ft/{task_name}_stderr": stderr})
                        except Exception as log_e:
                            logger.warning(f"Could not log zero-shot metric for {task_name}: {log_e}")
            except Exception as e:
                logger.error(f"Error during zero-shot evaluation: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("Zero-shot evaluation skipped (eval_zero_shot=False)")
        
        logger.info("=" * 60)
        logger.info("Post-training evaluation completed")
        logger.info("=" * 60)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # Define flags for absl
    # Model arguments
    # flags.DEFINE_string('model_name_or_path', '/home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/opt-125m_wanda/0.6', 'Path to pretrained model or model identifier from huggingface.co/models')
    # flags.DEFINE_string('config_name', 'facebook/opt-125m', 'Pretrained config name or path if not the same as model_name')

    flags.DEFINE_string('model_name_or_path', '/home/hyeondojang/log_teams/opt-model/elsa_iclr26/save/Llama-2-7b-hf_wanda/0.6', 'Path to pretrained model or model identifier from huggingface.co/models')
    flags.DEFINE_string('config_name', 'meta-llama/Llama-2-7b-hf', 'Pretrained config name or path if not the same as model_name')
    
    # LoRA arguments
    flags.DEFINE_integer('lora_r', 8, 'LoRA rank parameter')
    flags.DEFINE_integer('lora_alpha', 16, 'LoRA alpha parameter')
    flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout parameter')
    
    # Data arguments
    flags.DEFINE_string('dataset_name', 'c4', 'The name of the dataset to use')
    flags.DEFINE_integer('steps', 4096, 'Number of training steps (train_samples = batch_size * steps)')
    flags.DEFINE_integer('eval_samples', 4, 'Maximum number of evaluation samples')
    flags.DEFINE_integer('block_size', 2048, 'Optional input sequence length after tokenization')
    
    # Training arguments
    flags.DEFINE_string('output_dir', './lora_ft_output', 'Output directory for model checkpoints')
    flags.DEFINE_integer('per_device_train_batch_size', 2, 'Batch size per device for training (reduced for memory efficiency, use gradient accumulation)')
    flags.DEFINE_integer('per_device_eval_batch_size', 1, 'Batch size per device for evaluation')
    flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
    flags.DEFINE_integer('num_train_epochs', 1, 'Number of training epochs')
    flags.DEFINE_integer('batch_size', 8, 'Total batch size (per_device_batch_size * gradient_accumulation_steps)')
    flags.DEFINE_enum('lr_scheduler_type', 'linear', ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], 'Learning rate scheduler type')
    flags.DEFINE_integer('logging_steps', 1, 'Number of steps between logging')
    flags.DEFINE_integer('eval_steps', 100, 'Number of steps between evaluations')
    flags.DEFINE_bool('do_train', True, 'Whether to run training')
    flags.DEFINE_bool('do_eval', True, 'Whether to run evaluation')
    flags.DEFINE_bool('overwrite_output_dir', True, 'Whether to overwrite the output directory (default True to avoid PyTorch 2.6 checkpoint loading issues)')
    flags.DEFINE_integer('seed', 0, 'Random seed')
    
    # Wandb arguments
    flags.DEFINE_bool('wandb', True, 'Whether to use wandb for logging')
    flags.DEFINE_string('wandb_project', 'test', 'Wandb project name')
    
    # Evaluation arguments
    flags.DEFINE_string('data_path', None, 'Path to dataset (optional, for PPL evaluation)')
    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to run zero-shot evaluation after training')
    
    # Training optimization arguments
    flags.DEFINE_bool('gradient_checkpointing', False, 'Whether to use gradient checkpointing to save memory (slower but uses less memory)')
    
    # Mixed precision training (FP16/BF16)
    flags.DEFINE_enum('mixed_precision', 'bf16', ['fp16', 'bf16', 'none'], 'Mixed precision training: fp16, bf16, or none')
    
    # Model loading options (Wanda compatibility)
    flags.DEFINE_string('device_map', 'auto', 'Device map for model loading: auto, None, or specific device (same as Wanda code)')
    flags.DEFINE_bool('use_torch_compile', True, 'Whether to use torch.compile (same as Wanda code, default True)')
    
    # Use absl app.run
    app.run(main)