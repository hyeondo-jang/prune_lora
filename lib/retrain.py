
import torch
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from .optimizers import MaskedAdam
from .data import get_dataset
import os

@dataclass
class RetrainTrainingArguments(TrainingArguments):
    retrain_epochs: int = field(default=1, metadata={"help": "Number of epochs for retraining."})
    retrain_learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for the MaskedAdam optimizer."})
    retrain_batch_size: int = field(default=2, metadata={"help": "The batch size per device for retraining."})
    retrain_steps: int = field(default=100, metadata={"help": "The number of training steps for retraining."})

def retrain_model(args, model, tokenizer, device):
    """
    Retrains the pruned model using the Hugging Face Trainer and MaskedAdam optimizer.
    """
    
    retrain_args = RetrainTrainingArguments(
        output_dir="./retrain_output",
        num_train_epochs=args.retrain_epochs,
        learning_rate=args.retrain_learning_rate,
        per_device_train_batch_size=args.retrain_batch_size,
        max_steps=args.retrain_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    train_dataset = get_dataset(
        args.dataset,
        tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
    )

    optimizer = MaskedAdam(model.parameters(), lr=retrain_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=retrain_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None),
    )

    trainer.train()
