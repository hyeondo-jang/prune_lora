
import torch
import sys
from transformers import TrainingArguments, Trainer, TrainerCallback, default_data_collator
from dataclasses import dataclass, field
from .optimizers import MaskedAdam
from .data import get_dataset
import os
import torch.distributed as dist
from transformers.optimization import get_scheduler
from transformers.utils import is_sagemaker_mp_enabled
import math
from .trainer import Retrainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
try:
    from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training
except ImportError:
    try:
        from peft import prepare_model_for_int8_training
    except ImportError:
        try:
            from peft.tuners.lora import prepare_model_for_int8_training
        except ImportError:
            from peft.utils.peft_utils import prepare_model_for_int8_training
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

@dataclass
class RetrainTrainingArguments(TrainingArguments):
    retrain_learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for the MaskedAdam optimizer."})
    retrain_batch_size: int = field(default=2, metadata={"help": "The batch size per device for retraining."})
    retrain_steps: int = field(default=100, metadata={"help": "The number of training steps for retraining."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of gradient accumulation steps."})

def retrain_model(args, model, tokenizer, device):
    """
    Retrains the pruned model using the Hugging Face Trainer and MaskedAdam optimizer.
    """
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
    if model.dtype == torch.float16: ## upcast to fp32 for stability
        model = model.to(torch.float32)
    retrain_args = RetrainTrainingArguments(
        output_dir="./retrain_output",
        learning_rate=args.retrain_learning_rate,
        per_device_train_batch_size=args.retrain_batch_size,
        max_steps=args.retrain_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_accumulation_steps=args.retrain_gradient_accumulation_steps,
        bf16=True
    )

    num_train_samples = retrain_args.max_steps * retrain_args.per_device_train_batch_size * retrain_args.gradient_accumulation_steps * world_size

    train_dataset = get_dataset(
        args.dataset,
        tokenizer,
        nsamples=num_train_samples,
        seed=args.seed,
        seqlen=model.seqlen,
        data_path=args.data_path
    )

    trainer = Retrainer(
        model=model,
        args=retrain_args,
        train_dataset=train_dataset,
    )

    trainer.train()

def retrain_lora(args, model, tokenizer, device):
    """
    Retrains the pruned model using LoRA fine-tuning.
    Uses the pruned model directly (no reloading from disk).
    """
    from absl import logging as absl_logging
    logger = absl_logging
    
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Get LoRA parameters from args
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16)
    lora_dropout = getattr(args, 'lora_dropout', 0.05)
    
    # Get training parameters from args
    lora_learning_rate = getattr(args, 'lora_learning_rate', 1e-4)
    lora_batch_size = getattr(args, 'lora_batch_size', 8)
    lora_steps = getattr(args, 'lora_steps', 4096)
    lora_eval_steps = getattr(args, 'lora_eval_steps', 100)
    lora_logging_steps = getattr(args, 'lora_logging_steps', 1)
    mixed_precision = getattr(args, 'lora_mixed_precision', 'bf16')
    gradient_checkpointing = getattr(args, 'lora_gradient_checkpointing', False)
    use_torch_compile = getattr(args, 'lora_use_torch_compile', True)
    lora_dataset = getattr(args, 'lora_dataset', args.dataset if hasattr(args, 'dataset') else 'c4')
    
    # Calculate gradient accumulation steps
    per_device_batch_size = getattr(args, 'lora_per_device_batch_size', 2)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_per_device = per_device_batch_size * num_gpus
    gradient_accumulation_steps = max(1, lora_batch_size // effective_per_device)
    actual_batch_size = gradient_accumulation_steps * effective_per_device
    
    if local_rank in [-1, 0]:
        logger.info("--- Starting LoRA Fine-tuning Phase ---")
        logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Training: lr={lora_learning_rate}, steps={lora_steps}, batch_size={lora_batch_size}")
        logger.info(f"Effective batch size: {actual_batch_size} (per_device={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, num_gpus={num_gpus})")
    
    # Upcast to FP32 if using BF16 mixed precision (same as retrain.py)
    if mixed_precision == 'bf16' and model.dtype == torch.float16:
        model = model.to(torch.float32)
        torch.cuda.empty_cache()
        if local_rank in [-1, 0]:
            logger.info("Model upcasted to FP32 for BF16 mixed precision training")
    
    # Prepare model for LoRA training
    model = prepare_model_for_int8_training(model)
    
    # Configure LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    
    # Log LoRA parameter statistics
    if local_rank in [-1, 0]:
        # Count total model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Count base model parameters (excluding LoRA)
        base_params = 0
        lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_params += param.numel()
            else:
                base_params += param.numel()
        
        logger.info("=" * 60)
        logger.info("LoRA Parameter Statistics:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters (LoRA only): {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
        logger.info(f"  Non-trainable parameters (base model): {non_trainable_params:,} ({100 * non_trainable_params / total_params:.4f}%)")
        logger.info(f"  Base model parameters: {base_params:,}")
        logger.info(f"  LoRA adapter parameters: {lora_params:,} ({100 * lora_params / total_params:.4f}%)")
        logger.info("=" * 60)
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if local_rank in [-1, 0]:
                logger.info("Gradient checkpointing enabled")
    
    # Move model to device if needed
    if device is not None:
        model = model.to(device)
    
    # Prepare dataset
    num_train_samples = actual_batch_size * lora_steps
    train_dataset = get_dataset(
        lora_dataset,
        tokenizer,
        nsamples=num_train_samples,
        seed=args.seed if hasattr(args, 'seed') else 0,
        seqlen=model.seqlen if hasattr(model, 'seqlen') else 2048,
        data_path=args.data_path if hasattr(args, 'data_path') else None
    )
    
    eval_dataset = get_dataset(
        lora_dataset,
        tokenizer,
        nsamples=getattr(args, 'lora_eval_samples', 4),
        data_type="validation",
        seed=args.seed if hasattr(args, 'seed') else 0,
        seqlen=model.seqlen if hasattr(model, 'seqlen') else 2048,
        data_path=args.data_path if hasattr(args, 'data_path') else None
    )
    
    # Setup TrainingArguments
    training_args = TrainingArguments(
        output_dir="./lora_retrain_output",
        learning_rate=lora_learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,
        max_steps=lora_steps,
        logging_steps=lora_logging_steps,
        eval_steps=lora_eval_steps,
        evaluation_strategy="steps" if getattr(args, 'lora_do_eval', True) else "no",
        save_strategy="no",
        report_to=[],  # Disable automatic wandb, handle manually
        fp16=(mixed_precision == 'fp16'),
        bf16=(mixed_precision == 'bf16'),
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        overwrite_output_dir=True,
    )
    
    # Wandb logging callback
    if has_wandb:
        class WandbLoggingCallback(TrainerCallback):
            def __init__(self):
                self.enabled = hasattr(args, 'wandb') and args.wandb and local_rank in [-1, 0]
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if self.enabled and logs is not None:
                    wandb.log(logs, step=state.global_step)
        
        wandb_callback = WandbLoggingCallback() if (hasattr(args, 'wandb') and args.wandb) else None
        callbacks = [wandb_callback] if wandb_callback else []
    else:
        callbacks = []
    
    # Compute metrics for evaluation
    def compute_metrics(eval_preds):
        import numpy as np
        logits, labels = eval_preds
        
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        
        try:
            perplexity = math.exp(loss.item())
        except OverflowError:
            perplexity = float("inf")
        
        return {"perplexity": perplexity}
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if getattr(args, 'lora_do_eval', True) else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks if callbacks else None,
        compute_metrics=compute_metrics if getattr(args, 'lora_do_eval', True) else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if getattr(args, 'lora_do_eval', True) else None,
    )
    
    # Setup model state dict for LoRA
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    # Use torch.compile if enabled
    if use_torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        if local_rank in [-1, 0]:
            logger.info("Model compiled with torch.compile")
    
    # Train
    trainer.train()
    
    if local_rank in [-1, 0]:
        logger.info("LoRA fine-tuning finished")
