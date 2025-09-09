
import torch
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from .optimizers import MaskedAdam
from .data import get_dataset
import os
import torch.distributed as dist
from transformers.optimization import get_scheduler
from transformers.utils import is_sagemaker_mp_enabled
import math
from .trainer import Retrainer

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

    retrain_args = RetrainTrainingArguments(
        output_dir="./retrain_output",
        learning_rate=args.retrain_learning_rate,
        per_device_train_batch_size=args.retrain_batch_size,
        max_steps=args.retrain_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_accumulation_steps=args.retrain_gradient_accumulation_steps,
    )

    num_train_samples = retrain_args.max_steps * retrain_args.train_batch_size * retrain_args.gradient_accumulation_steps * world_size

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
