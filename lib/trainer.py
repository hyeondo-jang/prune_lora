# 기본 라이브러리
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from absl import logging
import wandb

# PyTorch 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.distributed as dist
import shutil

# 외부 라이브러리
import numpy as np
from packaging import version
import re

# Hugging Face Transformers 관련 (핵심 모듈)
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from accelerate.data_loader import skip_first_batches
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.trainer_pt_utils import find_batch_size, nested_detach, nested_numpify, distributed_concat
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, TrainOutput, denumpify_detensorize, has_length, speed_metrics
from transformers.utils import (
    is_apex_available, 
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_torch_xla_available,
    is_accelerate_available
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
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

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

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration
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
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
            
        super().__init__(*args, **kwargs)
        if self.is_world_process_zero():
            logger.info(f"ADMMTrainer initialized with loss_type='{self.args.loss_type}'")
        
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        self.penalty_scheduler = None
        self.sparsity_scheduler = None

    def allocate_nonuniform_sparsity(self):
        """
        Calculates and allocates non-uniform sparsity directly to the optimizer state.
        This version relies solely on parameter names for identification, ensuring
        compatibility with advanced distributed strategies like FSDP.
        """
        if not hasattr(self, "admm_param_names") or not self.admm_param_names:
            logger.warning("`admm_param_names` not found on trainer. Skipping adaptive sparsity allocation.")
            return

        if self.is_world_process_zero():
            logger.info(f'Calculating adaptive sparsity based on {self.args.admm_adaptive_sparsity_samples} samples...')

        # --- Step 1: Accumulate gradients ---
        train_dataloader = self.get_train_dataloader()
        samples_processed = 0
        batch_size = self.args.per_device_train_batch_size
        num_batches_needed = max(1, self.args.admm_adaptive_sparsity_samples // batch_size)
        
        batches_processed = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # ...
            self.training_step(self.model, self._prepare_inputs(batch))
            # ...
            batches_processed += 1

        # --- Step 2: Calculate block-level sensitivity scores using parameter names ---
        if self.is_world_process_zero():
            logger.info("--- Analyzing gradient sensitivity for adaptive sparsity ---")
        num_layers = self.model.config.num_hidden_layers
        block_scores = {i: {'score': 0.0, 'num_params': 0} for i in range(num_layers)}
        unwrapped_model = unwrap_model(self.model)

        for name, param in unwrapped_model.named_parameters():
            # Use parameter name for identification, not id(param)
            if not param.requires_grad or param.grad is None or name not in self.admm_param_names:
                continue
            
            match = re.search(r'\.layers\.(\d+)\.', name)
            if not match: continue
            
            layer_idx = int(match.group(1))
            if layer_idx not in block_scores: continue

            grad_data = param.grad.detach().cpu().abs()
            if batches_processed > 1: grad_data.div_(batches_processed)
            
            sensitivity = (grad_data * param.data.detach().cpu().abs()).mean()
            block_scores[layer_idx]['score'] += sensitivity
            block_scores[layer_idx]['num_params'] += 1

        avg_block_scores = {i: data['score'] / data['num_params'] if data['num_params'] > 0 else 0.0 for i, data in block_scores.items()}

        # --- Step 3: Distribute target sparsity based on scores ---
        scores_tensor = torch.tensor(list(avg_block_scores.values()), device='cpu')
        target_sparsity = self.args.sparsity_ratio
        final_block_sparsity = {}
        if scores_tensor.sum() == 0:
            logger.warning("All sensitivity scores are zero. Falling back to uniform sparsity.")
            final_block_sparsity = {i: target_sparsity for i in range(num_layers)}
        else:
            inverted_scores = 1.0 / (scores_tensor + 1e-12)
            normalized_scores = inverted_scores / inverted_scores.sum()
            block_sparsities_tensor = target_sparsity * len(scores_tensor) * normalized_scores
            final_block_sparsity = {i: min(0.99, max(0.0, block_sparsities_tensor[i].item())) for i in range(len(block_sparsities_tensor))}

        # --- Step 4: Update optimizer state by iterating through model parameters ---
        unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
        
        if self.is_world_process_zero():
            logger.info("Allocating adaptive sparsities to optimizer state...")

        # Iterate through the model's parameters again to update state
        for name, param in unwrapped_model.named_parameters():
            # Use name to check if the parameter is part of the ADMM group
            if name not in self.admm_param_names:
                continue
            
            # Ensure the state is initialized before updating
            unwrapped_optimizer._lazy_init_admm_state(param)

            match = re.search(r'\.layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                new_sparsity = final_block_sparsity.get(layer_idx, unwrapped_optimizer.sparsity)
                unwrapped_optimizer.state[param]['sparsity'] = new_sparsity
            else:
                unwrapped_optimizer.state[param]['sparsity'] = unwrapped_optimizer.sparsity

        # --- Step 5: Logging and cleanup ---
        if self.is_world_process_zero() and final_block_sparsity:
            logger.info("Finished allocating adaptive sparsities.")
            sorted_blocks = sorted(final_block_sparsity.items())
            sparsity_table = wandb.Table(columns=["Block Index", "Allocated Sparsity", "Average Score"])
            
            for layer_idx, sparsity in sorted_blocks:
                score = avg_block_scores.get(layer_idx, 0.0)
                logger.info(f"  - Block {layer_idx:02d} allocated sparsity: {sparsity:.4f}, score: {score:.4e}")
                sparsity_table.add_data(f"Block {layer_idx:02d}", sparsity, score)
            
            if self.args.wandb:
                wandb.log({
                    "adaptive_sparsity_chart": wandb.plot.bar(sparsity_table, "Block Index", "Allocated Sparsity", title="Allocated Sparsity per Block"),
                    "adaptive_score_chart": wandb.plot.bar(sparsity_table, "Block Index", "Average Score", title="Average Sensitivity Score per Block")
                }, step=self.state.global_step)

            wandb_metrics = {f"adaptive_sparsity/block_{idx:02d}": sparsity for idx, sparsity in final_block_sparsity.items()}
            self.log(wandb_metrics)
        
        # Clean up gradients and memory
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def create_optimizer(self):
        """
        Overrides the base method to create the ADMM optimizer.
        It now stores the names of ADMM parameters in `self.admm_param_names`
        to ensure robust identification in distributed environments.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        if self.optimizer is None:
            layers_to_prune = [nn.Linear]
            admm_layers = find_layers(opt_model, layers=layers_to_prune)
            
            self.admm_param_names = set()
            for name in admm_layers:
                if 'lm_head' in name:
                    continue
                if hasattr(admm_layers[name], 'weight') and admm_layers[name].weight.requires_grad:
                    full_param_name = f'{name}.weight'
                    self.admm_param_names.add(full_param_name)
            admm_param_list = []
            other_param_list = []
            for name,param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                if name in self.admm_param_names:
                    admm_param_list.append(param)
                else:
                    other_param_list.append(param)
            
            admm_param_group = {
                'params': admm_param_list,
                'name': 'weights',
                'admm': True,  # Mark this group for ADMM/SAFE logic
            }
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
        
            
            if self.args.is_safe:
                
                self.optimizer = SAFE(
                    param_groups,
                    projection_fn=projection,
                    alpha=self.args.admm_alpha,
                    lmda=self.args.admm_lmda, sparsity=self.args.sparsity_ratio, interval=self.args.admm_interval, 
                    lr=self.args.learning_rate, prune_n=self.args.prune_n, prune_m=self.args.prune_m, 
                    base_optimizer=base_optimizer, rho = self.args.rho, comparison_group=self.args.admm_projection_comparison_group,
                    **base_optimizer_kwargs,
                )
            else:
                # Debug: print some param_names
                if self.is_world_process_zero():
                    print(f"Created param_names with {len(self.admm_param_names)} entries")
                    print(f"Sample param_names: {list(self.admm_param_names)[:3]}")

                self.optimizer = ADMM(
                    param_groups,
                    projection_fn= projection,
                    alpha=self.args.admm_alpha,  # Over-relaxation parameter
                    lmda=self.args.admm_lmda, sparsity=self.args.sparsity_ratio, interval=self.args.admm_interval, 
                    lr=self.args.learning_rate, prune_n=self.args.prune_n, prune_m=self.args.prune_m, comparison_group=self.args.admm_projection_comparison_group,
                    projection_mode= self.args.admm_projection_mode,
                    importance_ema=self.args.admm_importance_ema,
                    base_optimizer = base_optimizer,
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
                with torch.no_grad():
                    primal_residual = torch.sqrt(torch.sum(torch.stack([torch.norm(w.detach() - self.optimizer.state[w]['split'].detach())**2 for w in weight])))
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
        unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer

        for group in groups:
            if group.get('admm', False):
                weights = group['params']
                lmda = group.get('lmda', self.args.admm_lmda)
                dual_residual = 0.0
                for w in weights:
                    u = self.optimizer.state[w]['dual']
                    z = self.optimizer.state[w]['split']
                    p_sparsity = self.optimizer.state[w]['sparsity'] ## TODO: impelment importance 
                    z_new = projection([w+u], sparsity=p_sparsity, prune_n=unwrapped_optimizer.prune_n, prune_m=unwrapped_optimizer.prune_m, comparison_group=unwrapped_optimizer.comparison_group)[0]
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

    def get_tp_size(self) -> int:
        """Get the tensor parallel size from either the model or DeepSpeed config."""

        # 1. Check model.tp_size first
        if (model_tp := getattr(self.model, "_tp_size", None)) is not None:
            return model_tp

        # 2. Fall back to DeepSpeed config if enabled
        if self.is_deepspeed_enabled and (deepspeed_config := getattr(self.args, "hf_deepspeed_config", None)):
            return deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)

        # 3. Default fallback
        return 1

    def get_total_train_batch_size(self, args) -> int:
        """Calculates total batch size (micro_batch * grad_accum * dp_world_size).

        Note: Only considers DP and TP (dp_world_size = world_size // tp_size)."""
        dp_world_size = args.world_size // self.get_tp_size()
        return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size

    ## TODO: implement multi-gpu support!
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # --- Setup code (dataloader, steps, optimizer, scheduler, state, resume logic) ---
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled or self.is_tp_enabled
        )
        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    if delay_optimizer_creation:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        # self._load_scaler(resume_from_checkpoint)
        # self._load_admm_extra_vars(resume_from_checkpoint) # Keep disabled
        if self.args.admm_adaptive_sparsity:
            self.allocate_nonuniform_sparsity()

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

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
        # --- Main Training Loop ---
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

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

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if hooks:
                    for h in hooks: h.remove()
                    del hooks

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )
                ### gradient normalization / clipping
                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm
                    
                    unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
                    is_dual_update_step = (unwrapped_optimizer.current_step + 1) % unwrapped_optimizer.interval == 0

                    ## TODO: fix here to be compatible with stateful admm optimizer
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
                                unwrapped_optimizer.update_importance_matrix(vectors_on_device)
                                del vectors_on_device
                            del self.importance_accumulator
                            torch.cuda.empty_cache()
                        
                    if is_dual_update_step:
                        mask_diff = unwrapped_optimizer.get_mask_diff()
                        if self.is_world_process_zero():
                            logger.info(f'ADMM Step {self.state.global_step}: Mask difference: {mask_diff:.4f}')
                            wandb_metrics = {'ADMM_mask_difference': mask_diff}  
                            self.log(wandb_metrics)

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()
                    ##TODO: implement SAFE like optiimzer later
                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()
                        self.sparsity_scheduler.step()
                        self.penalty_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # --- End Batch Loop ---
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm=None, model=model, trial=trial, epoch=epoch, ignore_keys_for_eval=ignore_keys_for_eval)
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        # --- End Epoch Loop ---

        if self.is_world_process_zero():
            logger.info("\n\nADMM Training completed.\n\n")

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        # Final projection must be called on all processes to keep weights in sync
        unwrapped_optimizer.final_projection() 
        if self.is_world_process_zero():
            logger.info('final projection finished')

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()


        # Final loss calculation and metrics
        # Calculate average primary training loss (tr_loss contains sum of scaled losses)
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
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
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

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


    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader, total_train_batch_size: int
    ):
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `epoch_based`
        - `len_dataloader`
        - `max_steps`
        """
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )

