import numpy as np
import torch
from transformers import AutoTokenizer
from lib.prune import prune_safe, prune_alps, prune_wanda, prune_magnitude, prune_sparsegpt, prune_admm, globalprune_admm
from lib.retrain import retrain_model, retrain_lora
from lib.eval import eval_ppl, eval_zero_shot
from lib.data import get_loaders
from lib.utils import check_sparsity, get_llm, start_record_memory_history, stop_record_memory_history, export_memory_snapshot, compute_dense_outputs, calculate_reconstruction_error
from absl import logging, app, flags
from importlib.metadata import version
from argparse import Namespace
import os # os 모듈 import
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import StateDictOptions,get_model_state_dict
import torch.distributed as dist
import logging as stdlogging
import wandb

logging.info(f"{version('torch')=}")
logging.info(f"{version('transformers')=}")
logging.info(f"{version('accelerate')=}")
logging.info(f'# of gpus: {torch.cuda.device_count()}')

FLAGS = flags.FLAGS


def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict() 
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')

    if FLAGS.wandb and local_rank == 0:
        wandb.init(project=FLAGS.wandb_project)

        if not dict(wandb.config):  
            wandb.config.update(arguments)  
        else: 
            updated_args = {
                k: wandb.config.get(k, v) for k, v in arguments.items()
            }
            FLAGS = Namespace(**updated_args)
            logging.info(f"Updated args with wandb.config: {FLAGS}")
    else:
        if local_rank == 0:
            logging.info('\n' + '\n'.join([f'{k} = {v}' for k, v in arguments.items()]))
        
    
    # Setting seeds for reproducibility
    np.random.seed(FLAGS.seed)
    torch.random.manual_seed(FLAGS.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if FLAGS.sparsity_type != "unstructured":
        assert FLAGS.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, FLAGS.sparsity_type.split(":"))

    if local_rank == 0:
        logging.info(f"loading llm model {FLAGS.model}")

    model = get_llm(FLAGS.model, FLAGS.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if FLAGS.calculate_global_recon and local_rank == 0:
        logging.info('capturing dense output for global recon error')
        _ ,valdata = get_loaders(FLAGS.dataset, FLAGS.nsamples, seed=FLAGS.seed, tokenizer=tokenizer, seqlen=model.seqlen, data_path=FLAGS.data_path)
        dense_outs = compute_dense_outputs(valdata, model, FLAGS, device, dataset_type="validation", batch_size=2)
        logging.info('dense output capture done')
    if FLAGS.prune_method != 'global_admm':
        if FLAGS.prune_method == "magnitude":
            model = model.to(torch.float32)
        else:
            model = model.to(torch.bfloat16)
        model = model.to('cpu')
        # model = model.to(device)
    else:
        model = model.to('cpu')
        model.config.use_cache = False
    
    logging.info(f"Process {local_rank} uses device {device}")
    if local_rank == 0 and FLAGS.visualize_memory:
        start_record_memory_history()
    is_local_pruning = FLAGS.prune_method != 'global_admm' and FLAGS.prune_method != 'dense'
    if FLAGS.sparsity_ratio != 0:

        logging.info("pruning starts")

        # --- Local Pruning on Main Process Only ---
        if is_local_pruning and local_rank == 0:
            if FLAGS.prune_method == "wanda":
                prune_wanda(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif FLAGS.prune_method == "magnitude":
                prune_magnitude(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif FLAGS.prune_method == "sparsegpt":
                prune_sparsegpt(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif FLAGS.prune_method == "safe":
                prune_safe(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif FLAGS.prune_method == "alps":
                prune_alps(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        # --- Global Pruning (Distributed) ---
        elif FLAGS.prune_method == 'global_admm':
            globalprune_admm(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == 'dense':
            logging.info("No pruning applied, model remains dense.")
    
    if local_rank == 0 and FLAGS.visualize_memory:
        path = f'{FLAGS.model.split("/")[-1]}_{FLAGS.prune_method}_dual:{FLAGS.admm_dual_dtype}_split:{FLAGS.admm_split_dtype}_base_opt:{FLAGS.admm_base_optimizer}'
        export_memory_snapshot(path)
        stop_record_memory_history()
        torch.cuda.memory._record_memory_history(enabled=None)
        exit()
    
    if local_rank == 0:
        logging.info("Pruning finished")
    # ## sparsity sanity check after pruning (per-device)
    # logging.info("*"*30)
    # sparsity_ratio = check_sparsity(model,log_by_block=True)
    # logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    # logging.info("*"*30)
    if FLAGS.calculate_global_recon and local_rank == 0:
        model = model.to(torch.float32) ## upcast to fp32 for stability
        model = model.to(device)
        logging.info('calculating global reconstruction error')
        recon_error = calculate_reconstruction_error(model, dense_outs, FLAGS, device, batch_size=2)
        logging.info(f'global reconstruction error (MSE): {recon_error:.6f}')
        if FLAGS.wandb:
            wandb.log({"global_recon_error": recon_error})
        del dense_outs
        torch.cuda.empty_cache()
    if is_local_pruning and is_distributed: ## broadcast
        model = model.to(device) ## cpu -> gpu for nccl communication
        model.config.use_cache = False
        if local_rank == 0:
            logging.info("[rank0] broadcasting pruned weights/buffers")
        with torch.no_grad():
            for p in model.parameters():
                if not p.is_cuda:
                    p.data = p.data.to(device)
                dist.broadcast(p.data, src=0)
            for b in model.buffers():
                if not b.is_cuda:
                    b.data = b.data.to(device)
                dist.broadcast(b.data, src=0)
        dist.barrier()
    if local_rank == 0:
        logging.info("Pruning and synchronization finished")
    ### sparsity sanity check after broadcasting
    # logging.info("*"*30)
    # sparsity_ratio = check_sparsity(model,log_by_block=True)
    # logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    # logging.info("*"*30)

    if FLAGS.do_retrain:
        ## TODO: handle distributed retraining with gpa
        # if is_distributed and FLAGS.prune_method == 'global_admm': ## assume global_admm uses fsdp2 for distributed training.
        #     state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True,broadcast_from_rank0=True)
        #     full_state = get_model_state_dict(model, options=state_dict_options)
        #     del model
        #     torch.cuda.empty_cache()
        #     model = get_llm(FLAGS.model, FLAGS.seqlen)
        #     with torch.no_grad():
        #         _ = model.load_state_dict(full_state, strict=True)
        #     del full_state
        #     torch.cuda.empty_cache()
        #     model.to(device)
        
        if FLAGS.retrain_dataset is None:
            FLAGS.retrain_dataset = FLAGS.dataset
        if local_rank == 0:
            logging.info("--- Starting Retraining Phase ---")
        retrain_model(FLAGS, model, tokenizer, device)
        if local_rank == 0:
            logging.info("Retraining finished")

    if FLAGS.do_retrain_lora:
        if local_rank == 0:
            logging.info("--- Starting LoRA Fine-tuning Phase ---")
        retrain_lora(FLAGS, model, tokenizer, device)
        if local_rank == 0:
            logging.info("LoRA fine-tuning finished")

    if is_distributed:
        dist.barrier()
        state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = get_model_state_dict(model, options=state_dict_options)
        if local_rank == 0:
            model = get_llm(FLAGS.model, FLAGS.seqlen)
            model.load_state_dict(full_state)
        dist.destroy_process_group()

    if local_rank == 0:
        
        if "gemma-2-27b" in FLAGS.model:
            logging.info("gemma-2-27b model detected. Casting to torch.bfloat16 for stability.")
            model = model.to(torch.bfloat16)
        else:
            logging.info(f"Casting model ({FLAGS.model}) to torch.float16.")
            model = model.to(torch.float16)
        model.seqlen = FLAGS.seqlen
        model = model.to(device)
        model.eval()
        # sparsity sanity check
        logging.info("*"*30)
        sparsity_ratio = check_sparsity(model,log_by_block=True)
        logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
        logging.info("*"*30)
        
        # perplexity evaluation
        ppl_test = eval_ppl(FLAGS, model, tokenizer, device,data_path=FLAGS.data_path)
        logging.info([(key,ppl) for key,ppl in ppl_test.items()])
        if FLAGS.wandb:
            wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})
        ## zero-shot evaluation
        if FLAGS.eval_zero_shot:
            logging.info(f"--- Evaluating After Pruning with ({FLAGS.prune_method}, Zero-Shot) ---")
            accelerate = "70b" in FLAGS.model
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa", "piqa","race"]
            num_shot = 0
            results_after = eval_zero_shot(FLAGS, FLAGS.model, model, tokenizer, task_list, num_shot, accelerate)
            logging.info(f"Zero-shot results after pruning with ({FLAGS.prune_method}):")
            logging.info(results_after)
            if FLAGS.wandb:
                for task_name, metrics in results_after.items():
                    try:
                        acc = metrics.get('acc,none', metrics.get('acc', None))
                        stderr = metrics.get('acc_stderr,none', metrics.get('acc_stderr', None))
                        if acc is not None:
                                wandb.log({f"{FLAGS.prune_method}/{task_name}_acc": acc})
                        if stderr is not None:
                                wandb.log({f"{FLAGS.prune_method}/{task_name}_stderr": stderr})
                    except Exception as log_e:
                        logging.warning(f"Could not log zero-shot metric for {task_name}: {log_e}")
    


if __name__ == '__main__':
    flags.DEFINE_string('model', 'facebook/opt-125m', 'model to prune.')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model.')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.5, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('prune_method', "global_admm", ["magnitude", "wanda", "sparsegpt", "safe", "alps","global_admm", 'dense'], 'Pruning method.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2"], 'Calibration dataset.')
    # flags.DEFINE_string('data_path', '/home/kwanheelee/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2', 'Path to local raw dataset directory (e.g., ~/.cache/huggingface/hub/dataset). Overrides online download.')
    flags.DEFINE_string('data_path', None, 'Path to local raw dataset directory (e.g., ~/.cache/huggingface/hub/dataset). Overrides online download.')
    # SAFE hyperparams
    flags.DEFINE_float('lmda', 1e-3, 'Penalty parameter for SAFE dual update.')
    flags.DEFINE_integer('batch_size', 4, 'Batch size for SAFE.')
    flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for SAFE.')
    flags.DEFINE_integer('epochs', 30, 'Number of epochs for SAFE.')
    flags.DEFINE_integer('interval', 32, 'Dual update interval for SAFE.')
    flags.DEFINE_integer('warmup_epochs', 2, 'Warmup epochs for SAFE.')
    flags.DEFINE_integer('accumulation_steps', 1, 'Accumulation steps for SAFE')
    flags.DEFINE_bool('activation', False, 'Activation based sparsity, SAFE+.')
    flags.DEFINE_float('rho', 2e-4, 'Rho for SAM.')

    # Global ADMM hyperparams
    flags.DEFINE_float('admm_beta1', 0.9, 'Beta1 for ADMM Adam optimizer.')
    flags.DEFINE_float('admm_beta2', 0.95, 'Beta2 for ADMM Adam optimizer.')
    flags.DEFINE_integer('admm_num_train_samples', 4, 'Number of training samples for ADMM.')
    flags.DEFINE_integer('admm_num_eval_samples', 4, 'Number of evaluation samples for ADMM.')
    flags.DEFINE_bool('admm_save_inputs', False, 'Whether to save processed ADMM training inputs.')
    flags.DEFINE_string('admm_save_path', None , 'Path to save ADMM training results and checkpoints.')
    flags.DEFINE_bool('save_model',False, 'Whether to save the pruned model after ADMM training.')
    flags.DEFINE_bool('is_safe', False, 'Whether to use SAFE')
    
    # Training Loop Config
    flags.DEFINE_integer('admm_epochs', 1, 'Number of epochs for ADMM training.')
    flags.DEFINE_integer('admm_steps', 10, 'Max steps for ADMM training. Overrides admm_epochs if > 0.')
    flags.DEFINE_integer('admm_batch_size', 8, 'Batch size for ADMM training, per device.')
    flags.DEFINE_integer('admm_gradient_accumulation_steps', 1, 'Gradient accumulation steps for ADMM.')
    flags.DEFINE_bool('admm_gradient_checkpointing', True, 'Use gradient checkpointing for ADMM training. Set False when using FSDP')
    flags.DEFINE_float('admm_lr', 2e-4, 'Learning rate for ADMM base optimizer.')
    flags.DEFINE_string('admm_lr_scheduler', 'linear', 'Learning rate scheduler type for ADMM.')
    flags.DEFINE_integer('admm_warmup_steps', 0, 'Warmup steps for ADMM learning rate scheduler.')
    flags.DEFINE_float('admm_weight_decay', 0.0, 'Weight decay for ADMM base optimizer.')
    flags.DEFINE_enum('admm_precision', 'bf16', ['fp32', 'fp16', 'bf16'], 'Precision for ADMM training (fp16/bf16 enables Trainer autocast).')
    flags.DEFINE_enum('admm_projection_comparison_group', 'layer', ['layer','column', 'row'], 'Comparison group for ADMM projection (layer/column/row).')
    flags.DEFINE_enum('admm_projection_mode', 'identity', ['identity', 'activation', 'gradient','momentum', 'taylor'], 'Generalized projection mode for ADMM (identity/activation/gradient/momentum/taylor).')
    flags.DEFINE_bool('admm_projection_bias_correction',False,'Whether to use bias correction in ADMM projection (for momentum/taylor).')
    flags.DEFINE_float('admm_importance_ema', 0.00, 'EMA for importance in ADMM projection. If > 0, importance matrix is updated with EMA.')
    flags.DEFINE_float('admm_termination_threshold',1e-2,'Termination threshold for ADMM.')

    # ADMM Specific Config
    flags.DEFINE_float('admm_alpha', 1.0, 'Alpha parameter for ADMM over-relaxation. default is 1.0')
    flags.DEFINE_float('admm_lmda', 0.01, 'Lambda penalty parameter for ADMM (for constant schedule).')
    flags.DEFINE_float('admm_init_lmda', 0.0, 'Initial lambda value for ADMM scheduling.')
    flags.DEFINE_float('admm_final_lmda', 0.01, 'Final lambda value for ADMM scheduling.')
    flags.DEFINE_bool('admm_init_lambda_from_inv_resid', False, 'Initialize lambda from inverse of initial residual.')
    flags.DEFINE_float('admm_mu', 10.0, 'Mu parameter for Adaptive penalty ADMM')
    flags.DEFINE_float('admm_tau_incr', 2.0, 'Tau increase factor for Adaptive penalty ADMM')
    flags.DEFINE_float('admm_tau_decr', 2.0, 'Tau decrease factor for Adaptive penalty ADMM')
    flags.DEFINE_enum('admm_lmda_schedule_mode', 'constant', ['constant','linear','exponential','cosine', 'adaptive_boyd'], 'Mode for lambda schedule (e.g., linear, cosine).')
    flags.DEFINE_enum('admm_sparsity_schedule_mode', 'constant', ['constant','linear','exponential','cosine','cubic','cubic'], 'Mode for sparsity schedule (e.g., cubic, linear).')
    flags.DEFINE_float('admm_peak_sparsity_step', 1, 'Step ratio (0-1) at which peak sparsity is reached in ADMM training. e.g. 0.3 means peak sparsity is reached at step*0.3 (used only when admm_sparsity_schedule_mode is not constant)')
    flags.DEFINE_integer('admm_interval', 2, 'Interval for ADMM projection and dual updates.')
    flags.DEFINE_enum('admm_base_optimizer', 'adam', ['adam','adamw','adam8bit','adam4bit','sgd'], 'Base optimizer for ADMM primal update.')
    flags.DEFINE_bool('admm_blockwise_projection', False, 'Use blockwise projection in ADMM.')
    flags.DEFINE_bool('admm_activation_aware', False, 'Use activation-aware projection in ADMM.')
    flags.DEFINE_bool('admm_decouple', False, 'Decouple proximal update in ADMM (for Adam).')
    flags.DEFINE_enum('admm_dual_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM dual variable (fp32 or bf16).')
    flags.DEFINE_enum('admm_split_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM split variable (fp32 or bf16).')
    flags.DEFINE_enum('loss_type', 'ntp', ['rem', 'ntp'], "Loss type for ADMM training ('rem' for reconstruction, 'ntp' for next token prediction).")
    flags.DEFINE_bool('normalize_grad', False, 'Whether to normalize gradients during ADMM training. Note that gradient normalization is only performed with respect to the gradients of the training objective.')
    flags.DEFINE_bool('normalize_prox_grad', False, 'Whether to normalize the proximal gradient in ADMM.')
    flags.DEFINE_bool('mean_prox_grad',False, 'mean proximal grad for scale invariance')
    flags.DEFINE_bool('admm_nonuniform_sparsity', False, 'Whether to use non-uniform sparsity based on sensitivity scores in ADMM.')
    flags.DEFINE_string('admm_nonuniform_sparsity_config_file', None, 'Path to non-uniform sparsity configuration file (JSON format).')
    
    

    # ADMM early stop
    flags.DEFINE_bool('admm_early_stop', False, 'Whether to use early stopping during ADMM training.')
    flags.DEFINE_integer('admm_early_stopping_patience', 3, 'Patience epochs for early stopping during ADMM training')
    flags.DEFINE_float('admm_early_stopping_threshold', 0.01, 'Threshold for early stopping during ADMM training.')

    # Retraining
    flags.DEFINE_bool('do_retrain', False, 'Whether to perform a retraining phase after pruning.')
    flags.DEFINE_string('retrain_dataset', None, 'Dataset for retraining.')
    flags.DEFINE_float('retrain_learning_rate', 2e-5, 'Learning rate for the MaskedAdam optimizer.')
    flags.DEFINE_integer('retrain_batch_size', 2, 'The batch size per device for retraining.')
    flags.DEFINE_integer('retrain_steps', 1, 'The number of training steps for retraining.')
    flags.DEFINE_integer('retrain_gradient_accumulation_steps', 1, 'Gradient accumulation steps for retraining.')
    
    # LoRA fine-tuning
    flags.DEFINE_bool('do_retrain_lora', True, 'Whether to perform LoRA fine-tuning after pruning.')
    flags.DEFINE_string('lora_dataset', 'c4', 'Dataset for LoRA fine-tuning (defaults to retrain_dataset or dataset).')
    flags.DEFINE_float('lora_learning_rate', 1e-4, 'Learning rate for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_batch_size', 8, 'Total effective batch size for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_per_device_batch_size', 2, 'Per device batch size for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_steps', 100, 'Number of training steps for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_eval_steps', 4, 'Number of steps between evaluations for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_logging_steps', 1, 'Number of steps between logging for LoRA fine-tuning.')
    flags.DEFINE_integer('lora_eval_samples', 4, 'Number of evaluation samples for LoRA fine-tuning.')
    flags.DEFINE_bool('lora_do_eval', True, 'Whether to run evaluation during LoRA fine-tuning.')
    flags.DEFINE_integer('lora_r', 8, 'LoRA rank parameter.')
    flags.DEFINE_integer('lora_alpha', 16, 'LoRA alpha parameter.')
    flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout parameter.')
    flags.DEFINE_enum('lora_mixed_precision', 'bf16', ['fp16', 'bf16', 'none'], 'Mixed precision for LoRA fine-tuning.')
    flags.DEFINE_bool('lora_gradient_checkpointing', False, 'Whether to use gradient checkpointing for LoRA fine-tuning.')
    flags.DEFINE_bool('lora_use_torch_compile', True, 'Whether to use torch.compile for LoRA fine-tuning.')
    
    # Logging & Evaluation
    flags.DEFINE_integer('admm_logging_steps', 1, 'Logging step interval for ADMM training.')
    flags.DEFINE_integer('admm_eval_steps', 1, 'Evaluation step interval for ADMM training.')

    flags.DEFINE_bool('data_ablation', False, 'Whether to use data ablation, for section 5.5. If True, we fix the step size and control the number of train samples with --admm_num_train_samples.')
    flags.DEFINE_bool('calculate_global_recon', False, 'Whether to calculate global reconstruction error.')
    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to evaluate zero-shot performance.')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', 'safe-torch', 'wandb project name.')
    flags.DEFINE_bool('visualize_memory', False, 'Whether to visualize memory usage.')
    app.run(main)
