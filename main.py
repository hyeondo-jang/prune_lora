import numpy as np
import torch
from transformers import AutoTokenizer
from lib.prune import prune_safe, prune_alps, prune_wanda, prune_magnitude, prune_sparsegpt, prune_admm, globalprune_admm
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity, get_llm
from absl import logging, app, flags
from importlib.metadata import version
from argparse import Namespace
import os # os 모듈 import

#logging
import wandb

logging.info(f"{version('torch')=}")
logging.info(f"{version('transformers')=}")
logging.info(f"{version('accelerate')=}")
logging.info(f'# of gpus: {torch.cuda.device_count()}')

FLAGS = flags.FLAGS


def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict() 
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ## delete afterwards
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
    model = get_llm(FLAGS.model,FLAGS.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)

    # torchrun/accelerate sets this environment variable
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if "30b" in FLAGS.model or "65b" in FLAGS.model:
        # For very large models with device_map, this logic might need careful handling.
        # The Trainer will handle device placement, so we can often rely on its logic.
        # For now, we assume the primary device is what matters for initial setup.
        device = model.hf_device_map["lm_head"]
    
    logging.info(f"Process {local_rank} uses device {device}")
    if FLAGS.sparsity_ratio != 0:
        logging.info("pruning starts")
        ### local pruners
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
        elif FLAGS.prune_method == 'global_admm':
            globalprune_admm(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == 'dense':
            logging.info("No pruning applied, model remains dense.")
        
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.barrier()
    
    if local_rank == 0:
        logging.info("Pruning finished on all processes. Starting evaluation on rank 0.")

    if local_rank == 0:
        # Move the final model to the designated device for evaluation
        model = model.to(torch.float16)
        model.seq_len = FLAGS.seqlen
        model = model.to(device)

        # sparsity sanity check
        logging.info("*"*30)
        sparsity_ratio = check_sparsity(model)
        logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
        logging.info("*"*30)
        
        # perplexity evaluation
        ppl_test = eval_ppl(FLAGS, model, tokenizer, device)
        logging.info([(key,ppl) for key,ppl in ppl_test.items()])
        if FLAGS.wandb:
            wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})

        ## zero-shot evaluation
        if FLAGS.eval_zero_shot:
            logging.info(f"--- Evaluating After Pruning with ({FLAGS.prune_method}, Zero-Shot) ---")
            accelerate = "70b" in FLAGS.model
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
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
    
    # Other processes will exit gracefully after the barrier.
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    flags.DEFINE_string('model', 'google/gemma-2-2b', 'model to prune.')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model.')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.5, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('prune_method', "global_admm", ["magnitude", "wanda", "sparsegpt", "safe", "alps","global_admm", 'dense'], 'Pruning method.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2"], 'Calibration dataset.')
    
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
    flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam.')
    flags.DEFINE_float('beta2', 0.95, 'Beta2 for Adam.')

    # Global ADMM hyperparams
    flags.DEFINE_integer('admm_num_train_samples', 4, 'Number of training samples for ADMM.')
    flags.DEFINE_integer('admm_num_eval_samples', 4, 'Number of evaluation samples for ADMM.')
    flags.DEFINE_bool('admm_save_inputs', False, 'Whether to save processed ADMM training inputs.')
    flags.DEFINE_string('admm_save_path', '', 'Path to save ADMM training results and checkpoints.')
    flags.DEFINE_bool('save_model',False, 'Whether to save the pruned model after ADMM training.')
    flags.DEFINE_bool('is_safe', False, 'Whether to use SAFE')
    

    # Training Loop Config
    flags.DEFINE_integer('admm_epochs', 1, 'Number of epochs for ADMM training.')
    flags.DEFINE_integer('admm_steps', 100, 'Max steps for ADMM training. Overrides admm_epochs if > 0.')
    flags.DEFINE_integer('admm_batch_size', 2, 'Batch size for ADMM training.')
    flags.DEFINE_integer('admm_gradient_accumulation_steps', 4, 'Gradient accumulation steps for ADMM.')
    flags.DEFINE_bool('admm_gradient_checkpointing', True, 'Use gradient checkpointing for ADMM training.')
    flags.DEFINE_float('admm_lr', 1e-4, 'Learning rate for ADMM base optimizer.')
    flags.DEFINE_string('admm_lr_scheduler', 'linear', 'Learning rate scheduler type for ADMM.')
    flags.DEFINE_integer('admm_warmup_steps', 0, 'Warmup steps for ADMM learning rate scheduler.')
    flags.DEFINE_float('admm_weight_decay', 0.0, 'Weight decay for ADMM base optimizer.')
    flags.DEFINE_enum('admm_precision', 'bf16', ['fp32', 'fp16', 'bf16'], 'Precision for ADMM training (fp16/bf16 enables Trainer autocast).')
    flags.DEFINE_enum('admm_projection_comparison_group', 'layer', ['layer','column', 'row'], 'Comparison group for ADMM projection (layer/column/row).')
    
    # ADMM Specific Config
    flags.DEFINE_float('admm_lmda', 0.1, 'Lambda penalty parameter for ADMM.')
    flags.DEFINE_float('admm_initial_lmda',0.0, 'Initial lambda value for ADMM (if using schedule).')
    flags.DEFINE_enum('admm_lmda_schedule_mode', 'constant', ['constant','linear','log','cosine'], 'Mode for lambda schedule (e.g., linear, cosine).')
    flags.DEFINE_enum('admm_sparsity_schedule_mode', 'constant', ['constant','linear','exponential','cosine','cubic','cubic'], 'Mode for sparsity schedule (e.g., cubic, linear).')
    flags.DEFINE_float('admm_peak_sparsity_step', 1, 'Step ratio (0-1) at which peak sparsity is reached in ADMM training. e.g. 0.3 means peak sparsity is reached at step*0.3 (used only when admm_sparsity_schedule_mode is not constant)')
    flags.DEFINE_integer('admm_interval', 32, 'Interval for ADMM projection and dual updates.')
    flags.DEFINE_enum('admm_base_optimizer', 'adam', ['adam', 'sgd'], 'Base optimizer for ADMM primal update.')
    flags.DEFINE_bool('admm_blockwise_projection', False, 'Use blockwise projection in ADMM.')
    flags.DEFINE_bool('admm_activation_aware', False, 'Use activation-aware projection in ADMM.')
    flags.DEFINE_bool('admm_decouple', False, 'Decouple proximal update in ADMM (for Adam).')
    flags.DEFINE_enum('loss_type', 'ntp', ['rem', 'ntp'], "Loss type for ADMM training ('rem' for reconstruction, 'ntp' for next token prediction).")

    # Logging & Evaluation
    flags.DEFINE_integer('admm_logging_steps', 10, 'Logging step interval for ADMM training.')
    flags.DEFINE_integer('admm_eval_steps', 100, 'Evaluation step interval for ADMM training.')



    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to evaluate zero-shot performance.')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', 'safe-torch', 'wandb project name.')
    
    app.run(main)
