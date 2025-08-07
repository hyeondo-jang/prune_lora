# 기본 라이브러리
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from absl import logging
import wandb
from tqdm import tqdm

# PyTorch 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 외부 라이브러리
import numpy as np
from packaging import version
import re

# Hugging Face Transformers 관련 (핵심 모듈)
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import find_batch_size, nested_detach, nested_numpify, distributed_concat
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, TrainOutput, denumpify_detensorize, has_length, speed_metrics
from transformers.utils import (
    is_apex_available, 
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

# 현재 프로젝트 관련
from .optimizers import ADMM,SAFE
from .scheduler import PenaltyScheduler, SparsityScheduler
from .utils import find_layers, projection

# =====================================================================================
# 조건부 Import 블록: 분산 환경 및 특정 라이브러리 의존성 처리
# 이 블록은 필요한 라이브러리가 없을 경우 오류를 발생시키지 않고 None으로 처리합니다.
# =====================================================================================
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

deepspeed_init = None
if is_sagemaker_mp_enabled():
    try:
        from transformers.integrations.deepspeed import deepspeed_init
    except ImportError:
        deepspeed_init = None

# TPU 관련
pl = None
nested_xla_mesh_reduce = None
if is_torch_tpu_available():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        from transformers.trainer_pt_utils import nested_xla_mesh_reduce
    except ImportError:
        pl = None
        nested_xla_mesh_reduce = None

# DDP/Deepspeed 관련
distributed_concat = None
try:
    from transformers.trainer_pt_utils import distributed_concat
except ImportError:
    distributed_concat = None

if is_apex_available():
    from apex import amp
# =====================================================================================

logger = logging
# logger = logging.get_logger(__name__)

# 체크포인트 파일 이름
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"



def compute_metrics(eval_preds: EvalPrediction):
    """Computes evaluation metrics, including perplexity."""
    # eval_preds.predictions = logits
    # eval_preds.label_ids = labels
    logits, labels = eval_preds.predictions, eval_preds.label_ids

    if logits is None or labels is None:
        logger.warning("compute_metrics: Logits or labels are None, cannot compute perplexity.")
        return {}

    # Shift so that tokens < n predict n
    # Convert to torch tensors if numpy arrays
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    # Flatten the tokens
    try:
        # Ensure logits and labels are float and long respectively, and on CPU for safety if needed
        shift_logits_float = shift_logits.view(-1, shift_logits.size(-1)).float() # Ensure float
        shift_labels_long = shift_labels.view(-1).long() # Ensure long

        # Move to CPU if tensors are large to avoid potential GPU OOM during calculation
        shift_logits_float = shift_logits_float.cpu()
        shift_labels_long = shift_labels_long.cpu()

        loss = loss_fct(shift_logits_float, shift_labels_long)
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
        loss = torch.tensor(float("inf")) # Assign inf loss as well
    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}", exc_info=True)
        perplexity = float("inf")
        loss = torch.tensor(float("inf"))

    return {"perplexity": perplexity, "eval_cross_entropy_loss": loss.item()}


class ADMMTrainer(Trainer):
    """
    Trainer using the external ADMM optimizer from lib.utils.optimizers.
    Supports standard Causal LM loss and Reconstruction Error Minimization (REM) loss.
    Includes Zeroth-Order (ZO) optimization capabilities.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
            
        super().__init__(*args, **kwargs)
        if self.is_world_process_zero():
            logger.info(f"ADMMTrainer initialized with loss_type='{self.args.loss_type}'")
        self.penalty_scheduler = None
        self.sparsity_scheduler = None

    def create_optimizer(self):
        """
        Overrides the base method to create the ADMM optimizer.
        - ADMM part only includes Linear weights and their duals/splits.
        - Base optimizer includes ALL trainable parameters, grouped by weight decay.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        param_sparsity_map = None
        if self.optimizer is None:
            layers_to_prune = [nn.Linear]
            admm_layers = find_layers(opt_model, layers=layers_to_prune)
            
            admm_param_list = []
            admm_param_ids = set()
            for name in admm_layers:
                if 'lm_head' in name:
                    continue
                if hasattr(admm_layers[name], 'weight') and admm_layers[name].weight.requires_grad:
                    admm_param_ids.add(id(admm_layers[name].weight))
                    admm_param_list.append(admm_layers[name].weight)
            admm_param_group = {
                'params': admm_param_list,
                'name': 'weights',
                'admm': True,  # Mark this group for ADMM/SAFE logic
            }
            other_param_list = []
            for param in opt_model.parameters():
                if param.requires_grad and id(param) not in admm_param_ids:
                    other_param_list.append(param)
            other_param_group = {
                'params': other_param_list,
                'name': 'other_params',
                'admm': False,  # Not for ADMM/SAFE logic
            }
            param_groups = []
            if admm_param_group['params']: # Only add if there are params in this group
                param_groups.append(admm_param_group)
            if other_param_group['params']: # Only add if there are params in this group
                param_groups.append(other_param_group)

            
            if not any(pg.get('admm', False) for pg in param_groups) and self.args.rank == 0:
                logger.warning(f'No parameters were marked for ADMM/SAFE. Pruning might not be effective.')
            
            if self.is_world_process_zero():
                logger.info(f'Found {len(admm_param_list)} / {len([p for p in opt_model.parameters()])} parameters for ADMM/SAFE.')
            
            # Base optimizer for ADMM (not for SAFE, as SAFE uses SAM which has its own base)
            base_optimizer_kwargs = {
                'weight_decay': self.args.weight_decay
            }
        
            if self.args.base_optimizer_type == 'adam':
                base_optimizer = torch.optim.Adam
            elif self.args.base_optimizer_type == 'sgd':
                base_optimizer = torch.optim.SGD
            else:
                NotImplementedError(f"Base optimizer type '{self.args.base_optimizer_type}' is not implemented. Supported: 'adam'.")
            # Prepare kwargs for ADMM
            # Add Adam specific args if base is adam
            if self.args.base_optimizer_type == 'adam':
                 base_optimizer_kwargs["betas"] = (self.args.adam_beta1, self.args.adam_beta2)
                 base_optimizer_kwargs["eps"] = self.args.adam_epsilon
            
            # Instantiate ADMM optimizer
            if getattr(self.args, 'admm_adaptive_sparsity', False):
                def admm_param_filter(name: str, p: nn.Parameter) -> bool:
                    return id(p) in admm_param_ids
                if self.is_world_process_zero():
                    logger.info(f'calculating adaptive sparsity based on {self.args.admm_adaptive_sparsity_samples} samples..')
                
                # Get train dataloader and calculate gradients on multiple batches
                train_dataloader = self.get_train_dataloader()
                train_dataloader_iterator = iter(train_dataloader)
                samples_processed = 0
                batch_size = self.args.per_device_train_batch_size
                
                num_batches_needed = max(1, self.args.admm_adaptive_sparsity_samples // batch_size)
                
                # Process multiple batches to accumulate gradients
                batches_processed = 0
                for _ in tqdm(range(num_batches_needed), desc="Calculating SNIP Scores for adaptive sparsity"):
                    batch = next(train_dataloader_iterator)
                    self.training_step(self.model, batch)
                    samples_processed += batch_size
                    batches_processed += 1

                param_sparsity_map, block_sparsity_map, avg_block_scores = self.calculate_adaptive_sparsity(
                    model=unwrap_model(self.model),
                    target_sparsity=self.args.sparsity_ratio,
                    num_layers=self.model.config.num_hidden_layers,
                    param_filter=admm_param_filter,
                    num_batches=batches_processed
                )
                if self.is_world_process_zero() and block_sparsity_map:
                    logger.info("Fixed adaptive sparsities for training.")
                    # Log the allocated sparsity for each block directly from the returned map
                    sorted_blocks = sorted(block_sparsity_map.items())
                    
                    # --- W&B Table Logging ---
                    # 1. Create a wandb.Table
                    sparsity_table = wandb.Table(columns=["Block Index", "Allocated Sparsity", "Average Score"])
                    
                    # 2. Populate the table
                    for layer_idx, sparsity in sorted_blocks:
                        score = avg_block_scores.get(layer_idx, 0.0)
                        logger.info(f"  - Block {layer_idx:02d} allocated sparsity: {sparsity:.4f}, score: {score:.4e}")
                        sparsity_table.add_data(f"Block {layer_idx:02d}", sparsity, score)
                    
                    # 3. Log the table with a custom bar chart by calling wandb.log() directly
                    if self.args.wandb:
                        if self.is_world_process_zero():
                            wandb.log({
                                "adaptive_sparsity_chart": wandb.plot.bar(
                                    sparsity_table, 
                                    "Block Index", 
                                    "Allocated Sparsity",
                                    title="Allocated Sparsity per Block"
                                ),
                                "adaptive_score_chart": wandb.plot.bar(
                                    sparsity_table,
                                    "Block Index",
                                    "Average Score",
                                    title="Average Sensitivity Score per Block"
                                )
                            }, step=self.state.global_step)
                        # --- End W&B Table Logging ---

                    # Keep individual logging for simple tracking if needed
                    wandb_metrics = {f"adaptive_sparsity/block_{idx:02d}": sparsity for idx, sparsity in block_sparsity_map.items()}
                    self.log(wandb_metrics)
                # 4. Zero out the gradients from the dry run
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache() 

            if self.args.is_safe:
                # Create parameter name mapping for the optimizer
                param_names = {}
                for name, param in self.model.named_parameters():
                    if admm_param_filter(name, param):
                        param_names[param] = name
                
                self.optimizer = SAFE(
                    param_groups,
                    projection_fn=projection,
                    alpha=self.args.admm_alpha,
                    lmda=self.args.admm_lmda, sparsity=self.args.sparsity_ratio, interval=self.args.admm_interval, 
                    lr=self.args.learning_rate, prune_n=self.args.prune_n, prune_m=self.args.prune_m, 
                    base_optimizer=base_optimizer, rho = self.args.rho, comparison_group=self.args.admm_projection_comparison_group,
                    param_names=param_names,  # Pass parameter name mapping
                    **base_optimizer_kwargs,
                )
            else:
                # Create parameter name mapping for the optimizer
                param_names = {}
                for name, param in self.model.named_parameters():
                    if admm_param_filter(name, param):
                        param_names[param] = name
                
                # Debug: print some param_names
                if self.is_world_process_zero():
                    print(f"Created param_names with {len(param_names)} entries")
                    print(f"Sample param_names: {list(param_names.values())[:3]}")
                
                self.optimizer = ADMM(
                    param_groups,
                    projection_fn= projection,
                    alpha=self.args.admm_alpha,  # Over-relaxation parameter
                    lmda=self.args.admm_lmda, sparsity=self.args.sparsity_ratio, param_sparsity_map = param_sparsity_map, interval=self.args.admm_interval, 
                    lr=self.args.learning_rate, prune_n=self.args.prune_n, prune_m=self.args.prune_m, comparison_group=self.args.admm_projection_comparison_group,
                    projection_mode= self.args.admm_projection_mode,
                    importance_ema=self.args.admm_importance_ema,
                    base_optimizer = base_optimizer,
                    param_names=param_names,  # Pass parameter name mapping
                    **base_optimizer_kwargs,
                )
            if self.is_world_process_zero():
                logger.info(f"Created {self.optimizer.__class__.__name__} optimizer targeting {len(admm_param_list)} Linear weight tensors.")
        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Override to create the ADMM optimizer and ensure the scheduler targets the base optimizer.
        If adaptive sparsity is enabled, it's calculated once before training starts.
        """
        self.create_optimizer()
        optimizer = self.optimizer

        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer.base_optimizer)
        if self.is_world_process_zero():
            logger.info("ADMM optimizer and scheduler created (scheduler targets base optimizer).")

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        creates a learning rate scheduler, penalty scheduler, and sparsity scheduler if needed.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        if self.penalty_scheduler is None:
            self.penalty_scheduler = PenaltyScheduler(
                optimizer=self.optimizer,
                initial_lmda=self.args.admm_initial_lmda if self.args.admm_initial_lmda >= 0 else self.args.admm_lmda,
                final_lmda= self.args.admm_lmda,
                total_steps=num_training_steps,
                mode=self.args.admm_lmda_schedule_mode
            )
        if self.sparsity_scheduler is None:
            if self.args.admm_adaptive_sparsity:
                NotImplementedError("Adaptive sparsity-aware scheduling is not implemented yet.")
            self.sparsity_scheduler = SparsityScheduler(
                optimizer=self.optimizer,
                initial_sparsity= 0.0,
                final_sparsity=self.args.sparsity_ratio,
                start_step=0,
                final_step=num_training_steps * self.args.admm_peak_sparsity_step,
                mode=self.args.admm_sparsity_schedule_mode
            )
        return self.lr_scheduler
    
    # --- Evaluation Methods (Standard implementations from Trainer, using the above prediction_step) ---
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step.
        - Handles REM loss calculation or standard loss.
        - Removes 'batch_idx' before calling model directly if not needed.
        - Returns generated standard labels when loss_type='rem' for metric computation.
        """
        has_labels = any(inputs.get(k) is not None for k in self.label_names)
        prepared_inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        original_labels = None
        if has_labels:
            original_labels = nested_detach(tuple(prepared_inputs.get(name) for name in self.label_names))
            if len(original_labels) == 1:
                original_labels = original_labels[0]

        loss = None
        logits = None
        labels_for_output = original_labels # Default to original labels

        with torch.inference_mode():
            with self.compute_loss_context_manager():
                if self.args.loss_type == "rem":
                    # --- REM Loss Path ---
                    # compute_loss returns (rem_loss, outputs) where outputs contains generated_labels_for_metrics
                    loss, outputs = self.compute_loss(model, prepared_inputs.copy(), return_outputs=True)
                    logits = outputs.get("logits")
                    # Use the generated standard labels for metrics when using REM loss
                    labels_for_output = outputs.get("generated_labels_for_metrics")
                    if labels_for_output is None:
                         logger.warning("REM loss used, but generated_labels_for_metrics not found in outputs.")
                         labels_for_output = original_labels # Fallback, though likely incorrect for PPL
                    # --- End REM Loss Path ---

                elif has_labels:
                    # --- Standard Loss Path (with labels) ---
                    inputs_for_std_loss = prepared_inputs.copy()
                    if 'batch_idx' in inputs_for_std_loss: inputs_for_std_loss.pop('batch_idx')
                    loss, outputs = self.compute_loss(model, inputs_for_std_loss, return_outputs=True)
                    logits = outputs.get("logits")
                    # labels_for_output remains original_labels
                    # --- End Standard Loss Path (with labels) ---

                else:
                    # --- No Labels Path ---
                    loss = None
                    inputs_for_fwd = prepared_inputs.copy()
                    if 'batch_idx' in inputs_for_fwd: inputs_for_fwd.pop('batch_idx')
                    outputs = model(**inputs_for_fwd)
                    logits = outputs.get("logits")
                    labels_for_output = None
                    # --- End No Labels Path ---

        # Detach outputs
        if loss is not None: loss = loss.mean().detach()
        if logits is not None: logits = nested_detach(logits)
        if labels_for_output is not None: labels_for_output = nested_detach(labels_for_output) # Detach generated labels too

        if prediction_loss_only:
            logits = None
            labels_for_output = None

        # Ensure tuple structure (loss, logits, labels)
        final_output = (loss, logits, labels_for_output)

        # Log loss if computed
        if loss is not None and self.state.is_world_process_zero:
             loss_key = "REM loss" if self.args.loss_type == "rem" else "CE loss"
             logger.debug(f"[prediction_step] Eval {loss_key}: {loss.item():.4f}")

        return final_output

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics. Standard implementation.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # Call evaluation_loop which uses self.prediction_step internally
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # Defer prediction_loss_only decision based on compute_metrics
            # prediction_loss_only=True if self.compute_metrics is None else None,
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Refactored to follow a more robust distributed evaluation pattern:
        1. Accumulate metrics locally on each process inside the loop.
        2. Perform a single synchronization (all_reduce) after the loop.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if not self.is_in_train:
            if args.fp16_full_eval: model = model.half()
            elif args.bf16_full_eval: model = model.bfloat16()

        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        if self.is_world_process_zero():
            logger.info(f"***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {num_examples}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        if args.past_index >= 0: self._past = None

        # Initialize local containers for accumulation
        losses_host_list = []
        total_ce_loss_sum = 0.0
        total_valid_tokens = 0
        observed_num_examples = 0

        # Main evaluation loop: accumulate locally
        for step, inputs in enumerate(dataloader):
            observed_num_examples += find_batch_size(inputs)
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host_list.append(losses)

            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                with torch.no_grad():
                    batch_ce_loss_sum = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    valid_tokens_mask = shift_labels != -100
                    batch_valid_tokens = valid_tokens_mask.sum().item()
                total_ce_loss_sum += batch_ce_loss_sum.item()
                total_valid_tokens += batch_valid_tokens

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        if args.past_index and hasattr(self, "_past"): delattr(self, "_past")

        # --- Synchronization and Metric Calculation (after the loop) ---
        metrics = {}

        # 1. Calculate Mean Original Loss (e.g., REM loss)
        if losses_host_list:
            losses_host = torch.cat(losses_host_list)
            if self.args.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
                losses_host = distributed_concat(losses_host)
            all_losses = nested_numpify(losses_host)
            if len(all_losses) > 0:
                loss_key = f"{metric_key_prefix}_rem_loss" if self.args.loss_type == "rem" else f"{metric_key_prefix}_loss"
                metrics[loss_key] = all_losses.mean().item()

        # 2. Synchronize and Calculate Perplexity
        # Use a robust check for distributed environment
        if self.args.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            total_ce_loss_sum_tensor = torch.tensor(total_ce_loss_sum, device=self.args.device)
            total_valid_tokens_tensor = torch.tensor(total_valid_tokens, device=self.args.device, dtype=torch.long)
            torch.distributed.all_reduce(total_ce_loss_sum_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_valid_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            total_ce_loss_sum = total_ce_loss_sum_tensor.item()
            total_valid_tokens = total_valid_tokens_tensor.item()

        if total_valid_tokens > 0:
            mean_ce_loss = total_ce_loss_sum / total_valid_tokens
            metrics[f"{metric_key_prefix}_ce_loss"] = mean_ce_loss
            try:
                metrics[f"{metric_key_prefix}_perplexity"] = math.exp(mean_ce_loss)
            except OverflowError:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        # # 3. Calculate W-Z Distance (only on the main process)
        if self.is_world_process_zero():
            primal_residual = self.calculate_primal_residual(
                metric_key_prefix=metric_key_prefix
            )
            dual_residual = self.calculate_dual_residual(
                metric_key_prefix=metric_key_prefix
            )
            metrics.update(primal_residual)
            metrics.update(dual_residual)

        metrics = denumpify_detensorize(metrics)
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=observed_num_examples)

    # --- End Evaluation Methods ---
    def calculate_primal_residual(
        self,
        metric_key_prefix: str = "eval",
        eps: float = 1e-12,          # small constant to prevent div-by-zero
    ) -> Dict[str, float]:
        """
        Compute both
            • primal residual ‖w - z‖₂
            • relative residual ‖w - z‖₂ / ‖w‖₂
        for all nn.Linear layers except lm_head.

        Returns
        -------
        dict
            {
            "<prefix>_primal_residual"  : float,
            "<prefix>_relative_residual": float
            }
        """
        groups = self.optimizer.param_groups
        for group in groups:
            if group.get('admm', False):
                weight = group['params']
                splits = group['splits']
                with torch.no_grad():
                    primal_residual = torch.sqrt(torch.sum(torch.stack([torch.norm(w.detach() - z.detach())**2 for w,z in zip(weight, splits)])))
                    relative_residual = primal_residual / (torch.sqrt(torch.sum(torch.stack([torch.norm(w.detach())**2 for w in weight]))) + eps)
        return {
            f"{metric_key_prefix}_primal_residual": primal_residual.item(),
            f"{metric_key_prefix}_relative_residual": relative_residual.item(),
        }
    def calculate_dual_residual(
        self,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Compute the dual residual for all nn.Linear layers except lm_head. (proj(w+u)-z)^2
        Returns
        -------
        dict
            {
            "<prefix>_dual_residual"  : float,
            "<prefix>_scaled_dual_residual": float
            }
        """
        groups = self.optimizer.param_groups
        for group in groups:
            if group.get('admm', False):
                weights = group['params']
                duals = group['duals']
                splits = group['splits']
                lmda = group.get('lmda', self.args.admm_lmda)
                dual_residual = 0.0
                for w,u,z in zip (weights, duals, splits):
                    z_new = projection([w+u], sparsity=self.optimizer.sparsity, prune_n=self.optimizer.prune_n, prune_m=self.optimizer.prune_m, comparison_group= self.optimizer.comparison_group)[0]
                    dual_residual += torch.norm(z_new-z, p=2)**2
                dual_residual = torch.sqrt(dual_residual)
        return {
            f"{metric_key_prefix}_dual_residual": dual_residual.item(),
            f"{metric_key_prefix}_scaled_dual_residual": lmda * dual_residual.item(),
        }
    
    # --- Training Methods ---

    def _compute_rem_loss_and_logits(self, model, inputs, return_outputs=False):
        """
        Helper to compute REM loss. Now gets REM labels directly from the input batch.
        """
        # --- FIX: Get REM labels from the input dictionary ---
        if 'rem_labels' not in inputs:
            raise ValueError("REM loss requires 'rem_labels' in the input batch.")
        
        target_labels_rem = inputs.pop("rem_labels")
        # --- END FIX ---

        if 'batch_idx' in inputs:
            inputs.pop("batch_idx") # batch_idx is no longer needed here

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        # logits = outputs.logits

        target_labels_rem = target_labels_rem.to(last_hidden_state.device, dtype=last_hidden_state.dtype)
        
        # Handle potential shape mismatch (e.g., due to different seqlen)
        seq_len_hidden = last_hidden_state.size(1)
        seq_len_target = target_labels_rem.size(1)
        if seq_len_hidden != seq_len_target:
             logger.warning(f"Shape mismatch! Hidden state seq len: {seq_len_hidden}, Target REM labels seq len: {seq_len_target}. Truncating.")
             seq_len = min(seq_len_hidden, seq_len_target)
             last_hidden_state = last_hidden_state[:, :seq_len, :]
             target_labels_rem = target_labels_rem[:, :seq_len, :]

        rem_loss = F.mse_loss(last_hidden_state, target_labels_rem)

        # --- Generate Standard Labels for Cross-Entropy/Perplexity ---
        generated_labels = None
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            generated_labels = input_ids.clone()
            generated_labels[:, -1] = -100 # Mask last token
        else:
            logger.warning("Cannot generate standard labels for perplexity calculation as 'input_ids' are missing.")

        if return_outputs:
            return rem_loss, outputs, generated_labels
        else:
            return rem_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes loss. The logic for REM loss now relies on 'rem_labels' being in the inputs.
        """
        if self.args.loss_type == "rem":
            # --- Reconstruction Error Loss Calculation (with optional CE regularization) ---
        
            # Always call the helper that returns logits and labels when return_outputs is needed
            # or when CE regularization is active (to calculate CE loss).
            if return_outputs:
                 # _compute_rem_loss_and_logits returns (rem_loss, outputs, generated_labels)
                 rem_loss, outputs, generated_labels = self._compute_rem_loss_and_logits(model, inputs, return_outputs=True)
                 logits = outputs.logits # Get logits from the outputs

                 # --- FIX: Add generated_labels to outputs for prediction_step ---
                 if hasattr(outputs, 'keys'): # Check if it's dict-like (e.g., ModelOutput)
                      outputs["generated_labels_for_metrics"] = generated_labels
                 else:
                      logger.warning("Outputs object is not dict-like, cannot add generated_labels_for_metrics.")
                 # --- END FIX ---
                 return (rem_loss, outputs)
            else: # return_outputs is False and ce_factor is 0.0
                 # Only REM loss is needed, and no outputs required
                rem_loss = self._compute_rem_loss_and_logits(model, inputs, return_outputs=False)
                return rem_loss
        else:
            # --- FIX: Standard loss path should also remove keys not meant for the model ---
            if 'batch_idx' in inputs:
                inputs.pop("batch_idx")
            if 'rem_labels' in inputs: # Just in case it's passed but not used
                inputs.pop("rem_labels")
            # --- END FIX ---
            return super().compute_loss(model, inputs, return_outputs=return_outputs)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a single training step.
        - Computes the primary loss (REM or CE).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        #Loss calculation
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        # --- Gradient Calculation and Scaling ---
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # backward
        if getattr(self,'do_grad_scaling',False):
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        del inputs
        torch.cuda.empty_cache()  # Clear cache to avoid OOM
        detached_loss = loss.detach()  # Detach the loss for logging and return
        return detached_loss
    
    ## TODO: implement multi-gpu support!
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # --- Setup code (dataloader, steps, optimizer, scheduler, state, resume logic) ---
        self._train_batch_size = batch_size
        args.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            # Use admm_steps or admm_epochs
            admm_max_steps = getattr(args, 'admm_steps', -1)
            admm_num_epochs = getattr(args, 'num_train_epochs', 5)
            if admm_max_steps > 0:
                max_steps = admm_max_steps
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(max_steps % num_update_steps_per_epoch > 0)
            else:
                max_steps = math.ceil(admm_num_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(admm_num_epochs)
            num_train_samples = max_steps * total_train_batch_size # Approx based on steps
        else:
             raise ValueError("ADMM training requires a dataloader with length.")

        if args.gradient_checkpointing:
            if self.is_world_process_zero():
                logger.info("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()

        self.model = self.model.to(args.device)

        # 2. Wrap the model for distributed training (DDP, etc.).
        #    Use self.model, not self.model_wrapped, as input.
        model = self._wrap_model(self.model, training=True, dataloader=train_dataloader)

        delay_optimizer_creation = (
             getattr(self, "sharded_ddp", None) is not None and getattr(self, "sharded_ddp", None) != "simple"
             or is_sagemaker_mp_enabled() or getattr(self, "fsdp", None) is not None
        )
        if not delay_optimizer_creation:
            if self.is_world_process_zero():
                logger.info(f"Creating ADMM optimizer and scheduler...")
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        # Use ADMM specific logging/eval steps
        self.state.logging_steps = getattr(args, 'admm_logging_steps', args.logging_steps)
        self.state.eval_steps = getattr(args, 'admm_eval_steps', args.eval_steps)


        # The following lines are for specific edge cases and should be kept
        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)
        
        # This is important to keep the internal state consistent after wrapping
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            if self.is_world_process_zero():
                logger.info(f"Delayed creation: Creating ADMM optimizer and scheduler...")
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        # self._load_admm_extra_vars(resume_from_checkpoint) # Keep disabled

        if self.is_world_process_zero():
            logger.info("***** Running ADMM Training *****")
            logger.info(f"  Num examples = {num_examples}")
            if args.max_steps > 0:
                logger.info(f"  ADMM Steps = {args.max_steps}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f'  Gradient Checkpointing = {args.gradient_checkpointing}')
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  ADMM Optimizer Args: lmda={args.admm_lmda}, interval={args.admm_interval}, sparsity={args.sparsity_ratio}")
            logger.info(f"  Loss Type = {self.args.loss_type}")
            logger.info(f'  ADMM Penalty Scheduler = {args.admm_lmda_schedule_mode}')
            logger.info(f'  ADMM Sparsity Scheduler = {args.admm_sparsity_schedule_mode}')


        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else: steps_trained_in_current_epoch = 0
            logger.info(f"  Continuing training from epoch {epochs_trained}, global step {self.state.global_step}")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
            
        # --- Main Training Loop ---
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1; continue

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                ## hook for activation aware generalized projection. use hook to catch micro-batch activation, accumulate importance vectors
                hooks = []
                if self.args.admm_projection_mode == 'activation':
                    is_first_microbatch = (step % self.args.gradient_accumulation_steps == 0)
                    if is_first_microbatch:
                        self.importance_accumulator = {}
                    
                    admm_param_ids = {id(p) for group in self.optimizer.param_groups if group.get('admm') for p in group['params']}
                    def create_hook(param_id, accumulator_dict):
                        @torch.no_grad()
                        def hook(module, inp, out)->None:
                            act_tensor = inp[0] if isinstance(inp, tuple) else inp
                            flat_act = act_tensor.reshape(-1,act_tensor.shape[-1])
                            diag = (flat_act **2).sum(dim=0).cpu()
                            if param_id not in accumulator_dict: accumulator_dict[param_id] = diag
                            else: accumulator_dict[param_id].add_(diag)
                            del act_tensor, flat_act, diag
                        return hook
                    
                    for module in model.modules():
                        if isinstance(module, nn.Linear) and id(module.weight) in admm_param_ids:
                            hooks.append(module.register_forward_hook(create_hook(id(module.weight), self.importance_accumulator)))

                tr_loss_step = self.training_step(model, inputs) # Now calls the simplified training_step

                if hooks:
                    for h in hooks: h.remove()
                    del hooks
                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    avg_scaled_loss = tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    tr_loss += avg_scaled_loss
                else:
                    tr_loss += tr_loss_step # Accumulate scaled loss

                self.current_flos += float(self.floating_point_ops(inputs))

                # --- Optimizer Step (Conditional on Accumulation) ---
                is_update_step = (step + 1) % args.gradient_accumulation_steps == 0 or \
                                (steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch)

                if is_update_step:
                    # --- First-Order Update ---
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        if getattr(self, 'do_grad_scaling', False): self.scaler.unscale_(self.optimizer)
                        if hasattr(self.optimizer, "clip_grad_norm"): self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"): model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            if args.normalize_grad: ## gradient normalization
                                grad_norm = nn.utils.get_total_norm(model.parameters())
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad.div_(grad_norm)
                            else:# default gradient clipping
                                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    is_dual_update_step = (self.optimizer.current_step + 1) % self.optimizer.interval == 0

                    ## activation generalized projection. reduce on micro-batch, update importance matrix
                    if self.args.admm_projection_mode == 'activation' and is_dual_update_step:

                        if hasattr(self, 'importance_accumulator') and self.importance_accumulator:
                            admm_params = next(g['params'] for g in self.optimizer.param_groups if g.get('admm'))
                            num_accumulation_steps = self.args.gradient_accumulation_steps
                            final_importance_vectors = [
                                (self.importance_accumulator.get(id(p)) / num_accumulation_steps) if self.importance_accumulator.get(id(p)) is not None else None
                                for p in admm_params
                            ]
                            if all(v is not None for v in final_importance_vectors):
                                device = admm_params[0].device
                                vectors_on_device = [v.to(device) for v in final_importance_vectors]
                                self.optimizer.update_importance_matrix(vectors_on_device)
                                del vectors_on_device
                            del self.importance_accumulator
                            torch.cuda.empty_cache()
                        
                    optimizer_was_run = True
                    if self.deepspeed: pass
                    elif getattr(self, 'do_grad_scaling', False):
                        scale_before = self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update(); scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        if isinstance(self.optimizer,SAFE):
                            self.optimizer.first_step(zero_grad=True) # Safe optimizer handles zero_grad internally
                            self.training_step(model, inputs) # Recompute loss for FO step
                            self.optimizer.second_step(zero_grad=True) # Second step for FO
                        else:
                            self.optimizer.step()

                    if is_dual_update_step:
                        mask_diff = self.optimizer.get_mask_diff()
                        if self.is_world_process_zero():
                            logger.info(f'ADMM Step {self.state.global_step}: Mask difference: {mask_diff:.4f}')
                            wandb_metrics = {'ADMM_mask_difference': mask_diff}  
                            self.log(wandb_metrics)

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()
                        self.sparsity_scheduler.step()
                        self.penalty_scheduler.step()
                    model.zero_grad() # Zero grad after FO step
    
                    # --- Common Update Step Logic ---
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm=None, model=model, trial=trial, epoch=epoch, ignore_keys_for_eval=ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop: break
            # --- End Batch Loop ---

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm=None, model=model, trial=trial, epoch=epoch, ignore_keys_for_eval=ignore_keys_for_eval)
            if self.control.should_training_stop: break
        # --- End Epoch Loop ---

        if self.is_world_process_zero():
            logger.info("\n\nADMM Training completed.\n\n")

        # Final projection must be called on all processes to keep weights in sync
        self.optimizer.final_projection()
        if self.is_world_process_zero():
            logger.info('final projection finished')

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.warning("load_best_model_at_end is True, which might overwrite the final projection if checkpoints were saved.")
            logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
            self._load_best_model()


        # Final loss calculation and metrics
        # Calculate average primary training loss (tr_loss contains sum of scaled losses)
        train_loss = self._total_loss_scalar / self.state.global_step if self.state.global_step > 0 else 0.0
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        # Add the primary training loss with a specific key
        if self.args.loss_type == 'rem':
            metrics["train_rem_loss"] = train_loss
        else:
            metrics["train_cross_entropy_loss"] = train_loss
        # Note: train_perplexity and train_cross_entropy_loss (for REM case) were logged during training steps

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics) # Log final training metrics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)


        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    @torch.no_grad()
    def prepare_inputs_for_tranformer_blocks(self, model, inputs, return_outputs=False):
        
        device = self.args.device
        
        class Catcher(nn.Module):
            """Catch hidden_states before first Transformer block (batch-aware)."""
            def __init__(self, module, cache, nsamples, storage):
                super().__init__()
                self.module = module
                self.cache = cache
                self.nsamples = nsamples
                self.storage = storage

            def forward(self, inp, **kwargs):
                hidden_states = inp[0] if isinstance(inp, tuple) else inp
                batch_size = hidden_states.size(0)

                end = self.cache['i'] + batch_size
                if end > self.nsamples:
                    raise RuntimeError(f"Catcher received too many samples: {end} > {self.nsamples}")

                self.storage[self.cache['i']:end] = hidden_states.detach()
                self.cache['i'] = end
                if self.cache['attention_mask'] is None and 'attention_mask' in kwargs:
                    if kwargs['attention_mask'] is not None:
                        self.cache['attention_mask'] = kwargs['attention_mask'].detach()
                if self.cache['position_ids'] is None and 'position_ids' in kwargs:
                    if kwargs['position_ids'] is not None:
                        self.cache['position_ids'] = kwargs['position_ids'].detach()

                raise ValueError  # Stop forward pass after capturing
        
        model.to('cpu')

        mtype = model.config.model_type

        if hasattr(model, 'model'): 
            if hasattr(model.model, 'decoder'):
                decoder = model.model.decoder 
            else: # gemma
                decoder = model.model
        elif hasattr(model, 'decoder'): # opt
            decoder = model.decoder
        else:
            raise ValueError(f"Unsupported model type: {mtype}")
        decoder.embed_tokens.to(device)

        if hasattr(decoder, 'embed_positions'):
            decoder.embed_positions.to(device)
        if hasattr(decoder, 'rotary_emb'):
            decoder.rotary_emb.to(device)
        decoder.layers[0].to(device)
        
        batch_size, seq_len = inputs["input_ids"].shape
        hidden_size = model.config.hidden_size
        nsamples = batch_size

        inps = torch.zeros(nsamples, seq_len, hidden_size, device=device)
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        original_block = decoder.layers[0]
        decoder.layers[0] = Catcher(original_block, cache, nsamples, inps)
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"])
        except ValueError:
            pass
        
        decoder.layers[0] = original_block
        decoder.embed_tokens.to("cpu")
        if hasattr(decoder, 'embed_positions'): 
            decoder.embed_positions.to("cpu")
        if hasattr(decoder, 'rotary_emb'):
            decoder.rotary_emb.to("cpu")
        decoder.layers[0].to("cpu")

        torch.cuda.empty_cache()

        return {k: v for k, v in {
            "inps": inps,
            "attention_mask": cache["attention_mask"],
            "position_ids":  cache["position_ids"],
        }.items() if v is not None}

    @torch.no_grad()
    def memory_efficient_compute_loss(self, model, inputs, return_outputs=False):
        
        model.eval()
        model.config.use_cache = False
        device = self.args.device

        layer_kwargs = self.prepare_inputs_for_tranformer_blocks(model, inputs)
        inps = layer_kwargs.pop('inps')

        if hasattr(model, 'model'): 
            if hasattr(model.model, 'decoder'):
                decoder = model.model.decoder 
            else: # gemma / llama
                decoder = model.model
        elif hasattr(model, 'decoder'): # opt
            decoder = model.decoder

        if hasattr(decoder, 'final_layer_norm'): # opt
            final_ln = decoder.final_layer_norm
        elif hasattr(decoder, 'norm'): # gemma
            final_ln = decoder.norm
        else:
            final_ln = None

        lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        
        for i, layer in enumerate(decoder.layers):
            layer = layer.to(device)
            inps = layer(inps, **layer_kwargs)[0]
            layer = layer.to('cpu')
            torch.cuda.empty_cache()
        
        del layer_kwargs
        
        if final_ln is not None:
            final_ln.to(device)
            inps = final_ln(inps)
            final_ln.to('cpu')
            torch.cuda.empty_cache()
        
        if lm_head is not None:
            lm_head = lm_head.to(device)
            logits = lm_head(inps)
            lm_head = lm_head.to('cpu')
            torch.cuda.empty_cache()
        else:
            logits = inps

        if getattr(model.config, "final_logit_softcapping", None) is not None:
            logits = logits / model.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * model.config.final_logit_softcapping
        
        labels = inputs.get("labels")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if return_outputs:
            return loss, logits
        else:
            return loss

    def calculate_adaptive_sparsity(
        self,
        model: nn.Module,
        target_sparsity: float,
        num_layers: int,
        param_filter: Callable[[str, nn.Parameter], bool] = lambda name, p: True,
        num_batches: int = 1,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calculates per-parameter sparsity based on sensitivity scores (|grad * weight|).
        No optimizer dependency.
        
        Args:
            model: the model with gradients computed.
            target_sparsity: overall target sparsity (e.g., 0.5)
            num_layers: number of decoder layers (e.g., 32 for LLaMA-7B)
            param_filter: optional filter function to select which parameters to include
        
        Returns:
            - param_sparsity_map: {param_name: sparsity value}
            - block_sparsity_map: {layer_idx: sparsity value}
        """
        param_sparsity_map = {}
        block_scores = {i: {'score': 0.0, 'num_params': 0} for i in range(num_layers)}
        
        # Step 1: Accumulate per-layer scores and log gradient distributions
        if self.is_world_process_zero():
            logger.info("--- Gradient Distribution Analysis ---")

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not param_filter(name, param):
                continue
            
            match = re.search(r'\d+', name)  # extract layer index from name
            if not match:
                continue
            
            layer_idx = int(match.group(0))
            if layer_idx not in block_scores:
                continue

            grad_data = param.grad.detach().cpu().abs()
            # Average the accumulated gradients by number of batches
            if num_batches > 1:
                grad_data = grad_data / num_batches
            sensitivity = (grad_data * param.data.detach().cpu().abs()).sum()
            block_scores[layer_idx]['score'] += sensitivity
            block_scores[layer_idx]['num_params'] += param.numel()

        # Step 2: Average block sensitivity
        avg_block_scores = {
            i: data['score'] / data['num_params'] if data['num_params'] > 0 else 0.0
            for i, data in block_scores.items()
        }
        scores_tensor = torch.tensor(list(avg_block_scores.values()), device='cpu')
        
        # Step 3: Inverse-score based sparsity allocation
        if self.args.admm_adaptive_sparsity_smooth:
            scores_tensor = torch.softmax(scores_tensor / self.args.admm_adaptive_sparsity_smooth_temperature, dim=0)
        inverted_scores = 1.0 / (scores_tensor + 1e-12)
        normalized_scores = inverted_scores / inverted_scores.sum()
        block_sparsities_tensor = target_sparsity * len(scores_tensor) * normalized_scores
        block_sparsity_map = {i: min(0.99, max(0.0, block_sparsities_tensor[i].item()))
                            for i in range(len(block_sparsities_tensor))}
        if self.is_world_process_zero():
            logger.info("--- Average Block Sensitivity Scores ---")
            for i, score in avg_block_scores.items():
                logger.info(f"Layer {i:02d}: Avg Score = {score:.4e}")
            logger.info("--- End Average Block Sensitivity Scores ---")
        # Step 4: Assign sparsity to each parameter
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not param_filter(name, param):
                continue
            
            match = re.search(r'\d+', name)
            if not match:
                param_sparsity_map[name] = target_sparsity
                continue

            layer_idx = int(match.group(0))
            sparsity = block_sparsity_map.get(layer_idx, target_sparsity)
            param_sparsity_map[name] = min(0.99, max(0.0, sparsity))

        torch.cuda.empty_cache()
        return param_sparsity_map, block_sparsity_map, avg_block_scores


