import time
import torch
import math 
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.amp import autocast
from .pruner import WrappedGPT, ALPS_prune, SparseGPT
from .data import get_loaders,get_dataset, TensorData,TensorDataLoader,TensorData, TensorDataLoader
from .optimizers import SAFE
from .utils import *
from .trainer import ADMMTrainer
# --- FIX: Import necessary dataset functions ---
from datasets import Dataset, concatenate_datasets
# --- END FIX ---
import torch
from absl import logging
from argparse import Namespace
from pathlib import Path
from datetime import datetime
import math
from dataclasses import dataclass, field # Import dataclass and field
import torch.nn.functional as F
# --- Necessary Imports ---
from transformers import TrainingArguments
try:
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
except ImportError:
    OPTDecoderLayer = None # Fallback if OPT is not available
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

## Local pruning
@torch.no_grad() 
def prune_magnitude(
    args,
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    device:torch.device,
    prune_n:int=0,
    prune_m:int=0
):
    """
    Prunes the model using the magnitude pruning (\|w\|) method.
    Removes weights with the smallest magnitudes, supporting unstructured or N:M structured sparsity.

    Args:
        args: Configuration object with attribute `sparsity_ratio (int)`.
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer (not directly used here but common signature).
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info("Starting magnitude pruning...")
    layers = get_model_layers(model)
 
    # Pruning based on magnitude
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer) 

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)

            if prune_n != 0 and prune_m != 0: # N:M structured sparsity
                W_mask = torch.zeros_like(W, dtype=torch.bool) 
                for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                    start_col = col_chunk_idx * prune_m
                    end_col = start_col + prune_m
                    tmp_metric_chunk = W_metric[:, start_col:end_col]
                    
                    _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)
                    
                    W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
            else: # Unstructured sparsity
                num_elements_to_prune = int(W.numel() * args.sparsity_ratio)
                threshold = torch.kthvalue(W_metric.flatten(), num_elements_to_prune + 1).values 
                W_mask = (W_metric <= threshold)

            W[W_mask] = 0 
        
        layers[i] = layer.to('cpu') 
        torch.cuda.empty_cache()
    logging.info("Magnitude pruning finished.")

@torch.no_grad() 
def prune_wanda(
    args,
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    device:torch.device,
    prune_n:int=0,
    prune_m:int=0
):
    """
    Prunes the model using the WANDA (\|w\|\|x\|_2) method.
    Original code: https://github.com/locuslab/wanda
    Args:
        args: Configuration object with `sparsity_ratio (int)`, `nsamples (int)`, `seed (int)`, `dataset ()`.
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer for loading calibration data.
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info("Starting WANDA pruning...")
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    logging.info("Loading calibration data for WANDA.")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    logging.info("Calibration data loaded.")
    with torch.no_grad():
        inps, outs_placeholder, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.nsamples)

    layers = get_model_layers(model)
    current_layer_outputs = torch.zeros_like(inps) 

    for i in range(len(layers)):
        logging.info(f"Pruning layer {i} with WANDA...")
        layer = layers[i].to(device)
        subset = find_layers(layer) 

        wrapped_layers = {} 
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch_hook(module_name):
            def hook_fn(_, inp, out):
                actual_input_tensor = inp[0] if isinstance(inp, tuple) else inp
                wrapped_layers[module_name].add_batch(actual_input_tensor.data, out.data)
            return hook_fn

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch_hook(name)))
        

        temp_dense_outputs = torch.zeros_like(inps) 
        for j in range(args.nsamples):
            with torch.no_grad():
                current_input_sample = inps[j].unsqueeze(0).to(device)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    temp_dense_outputs[j] = layer(current_input_sample, attention_mask=attention_mask)[0].to('cpu')
                else:
                    temp_dense_outputs[j] = layer(current_input_sample, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')
        
        for h in handles: # Remove hooks
            h.remove()

        # Prune weights based on WANDA metric
        for name in subset:
            logging.info(f"Applying WANDA to layer {i}, module {name}.")
            W = subset[name].weight.data
            # WANDA Metric: |W| * sqrt(Activation_Scale for each input neuron, i.e., row of W)
            # scaler_row corresponds to ||X_j||_2 / sqrt(num_samples) for each input feature j
            # So, W_metric_ij = |W_ij| * ||X_j||_2
            activation_scales = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(device) 
            W_metric = torch.abs(W) * activation_scales

            W_mask = torch.zeros_like(W, dtype=torch.bool) 
            if prune_n != 0 and prune_m != 0: # N:M structured sparsity
                for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                    start_col = col_chunk_idx * prune_m
                    end_col = start_col + prune_m
                    tmp_metric_chunk = W_metric[:, start_col:end_col]
                    _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)
                    W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
            else: # Unstructured sparsity
                num_cols_to_prune_per_row = int(W_metric.shape[1] * args.sparsity_ratio)
                if num_cols_to_prune_per_row > 0:
                    _, indices_to_prune = torch.topk(W_metric, num_cols_to_prune_per_row, dim=1, largest=False)
                    W_mask.scatter_(1, indices_to_prune, True)
            
            W[W_mask] = 0  # Apply pruning mask

        with torch.no_grad():
            for j in range(args.nsamples):
                current_input_sample = inps[j].unsqueeze(0).to(device)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    current_layer_outputs[j] = layer(current_input_sample, attention_mask=attention_mask)[0].to('cpu')
                else:
                    current_layer_outputs[j] = layer(current_input_sample, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')
        
        inps, current_layer_outputs = current_layer_outputs, inps #
        
        layers[i] = layer.to('cpu') # Offload layer
        del wrapped_layers # Free memory
        torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    logging.info("WANDA pruning finished.")

## TODO: need fix (projection, admm)
@torch.no_grad() 
def prune_safe(
    args,
    model:AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using SAFE method.
    Paper: https://dummy.com

    Args:
        args: Configuration object with learning_rate (float), lmda (float), rho (float), interval (int), epochs (int).
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer for loading calibration data.
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info('Starting SAFE pruning...')
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    use_cache = getattr(model.config, 'use_cache', None)
    
    if use_cache is not None:
        model.config.use_cache = False
    
    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.nsamples)

    layers = get_model_layers(model)

    current_layer_outputs = torch.zeros_like(inps) 
    logging.info('SAFE calibration data prepared.')

    # Pruning based on SAFE
    for i in range(len(layers)):
        layer = layers[i].float().to(device) 
        current_inps = inps.float().to(device) 
        
        learning_rate = args.learning_rate
        logging.info(f'Pruning layer {i} with SAFE. Learning rate: {learning_rate}')
        
        # Store outputs of the original (dense) layer to use as targets for optimization
        dense_layer_targets = torch.zeros_like(current_inps, device='cpu') 

        importance_matrix = None
        wrapped_submodules = {} # For collecting activations if args.activation is true
        
        if args.activation: 
            subset_for_act = find_layers(layer)
            importance_matrix = []
            for name in subset_for_act:
                wrapped_submodules[name] = WrappedGPT(subset_for_act[name])

            def add_batch_hook_act(module_name):
                def hook_fn(_, inp, out):
                    actual_input_tensor = inp[0] if isinstance(inp, tuple) else inp
                    wrapped_submodules[module_name].add_batch(actual_input_tensor.data, out.data)
                return hook_fn

            handles_act = []
            for name in wrapped_submodules:
                handles_act.append(subset_for_act[name].register_forward_hook(add_batch_hook_act(name)))
        
        # Get outputs from the dense (original) layer to serve as optimization targets
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0) 
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    dense_layer_targets[j] = layer(sample_input, attention_mask=attention_mask)[0].detach().to('cpu')
                else:
                    dense_layer_targets[j] = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0].detach().to('cpu')
        
        if args.activation: 
            for h in handles_act:
                h.remove() # Remove hooks

        # Prepare DataLoader for the optimization process
        tensordata = TensorData(current_inps, dense_layer_targets, device) 
        tensordata_loader = TensorDataLoader(tensordata, args.batch_size, shuffle=True, num_workers=0).get_loader()
        
        num_update_steps_per_epoch = len(tensordata_loader)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)
        warmup_steps = math.ceil(args.warmup_epochs * num_update_steps_per_epoch)

        if args.activation:
            for name in wrapped_submodules:
                act_scaler = torch.sqrt(wrapped_submodules[name].scaler_row.reshape((1, -1))).to(device)
                importance_matrix.append(act_scaler)
        
        prunable_submodules = find_layers(layer)
        # Collect parameters for the 'safe_params' group and their IDs
        safe_params_list = []
        safe_param_ids = set()
        for name in prunable_submodules:
            # Assuming only 'weight' is part of ADMM for now
            if hasattr(prunable_submodules[name], 'weight') and prunable_submodules[name].weight.requires_grad:
                weight_param = prunable_submodules[name].weight
                safe_params_list.append(weight_param)
                safe_param_ids.add(id(weight_param))


        safe_params_group = {
            'params': safe_params_list,
            'name': 'weights', # Or a more descriptive name like 'admm_weights'
            'admm': True,
        }
        # Collect all other parameters from the layer that are not in safe_params_list
        other_params_list = []
        for param in layer.parameters():
            if param.requires_grad and id(param) not in safe_param_ids:
                other_params_list.append(param)

        other_params_group = {
            'params': other_params_list,
            'name': 'other_params',
            'admm': False,
        }

        # Ensure param_groups only contains groups with actual parameters
        param_groups = []
        if safe_params_group['params']: # Only add if there are params in this group
            param_groups.append(safe_params_group)
        if other_params_group['params']: # Only add if there are params in this group
            param_groups.append(other_params_group)

        opt = SAFE(
            param_groups,
            projection_fn = projection,
            lmda=args.lmda, sparsity=args.sparsity_ratio, interval=args.interval, 
            lr=learning_rate, rho=args.rho, prune_n=prune_n, prune_m=prune_m, 
            importance_matrix=importance_matrix,
            betas=(args.beta1, args.beta2),
        )

        loss_fn = torch.nn.MSELoss().to(device)
        lr_scheduler = get_linear_schedule_with_warmup(
            opt.base_optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    
        global_step_counter = 0 
        sam_input_cache = []
        # Optimization loop
        for epoch in range(args.epochs): 
            epoch_start_time = time.time()
            total_epoch_loss_items = 0.0 
            
            for batch_inputs, batch_targets in tensordata_loader:
                
                with torch.enable_grad(): 
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                            outputs = layer(batch_inputs, attention_mask=attention_mask)[0]
                        else:
                            outputs = layer(batch_inputs, attention_mask=attention_mask, position_ids=position_ids)[0]
                        loss = loss_fn(outputs, batch_targets)
                        if args.accumulation_steps > 1:
                            loss = loss / args.accumulation_steps
                    loss.backward() 
                    global_step_counter += 1
                    if args.accumulation_steps > 1: 
                        sam_input_cache.append(batch_inputs.detach().clone())


                if global_step_counter % args.accumulation_steps == 0:
                    with torch.enable_grad(): 
                        opt.first_step(zero_grad=True) 
                        
                        if sam_input_cache: 
                            for cached_batch_input in sam_input_cache:
                                with autocast(device_type='cuda', dtype=torch.bfloat16):
                                    if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                                        loss_sam_step = loss_fn(layer(cached_batch_input, attention_mask=attention_mask)[0], batch_targets)
                                    else:
                                        loss_sam_step = loss_fn(layer(cached_batch_input, attention_mask=attention_mask, position_ids=position_ids)[0], batch_targets)
                                if args.accumulation_steps > 1:
                                     loss_sam_step = loss_sam_step / args.accumulation_steps
                                loss_sam_step.backward() 
                            sam_input_cache = [] 
                        else: 
                            if args.accumulation_steps == 1: 
                                with autocast(device_type='cuda', dtype=torch.bfloat16):
                                    if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                                        outputs_perturbed = layer(batch_inputs, attention_mask=attention_mask)[0]
                                    else:
                                        outputs_perturbed = layer(batch_inputs, attention_mask=attention_mask, position_ids=position_ids)[0]
                                    loss_perturbed = loss_fn(outputs_perturbed, batch_targets)
                                loss_perturbed.backward()


                        opt.second_step(zero_grad=True) 
                        
                    opt.zero_grad(set_to_none=True) 
                
                    total_epoch_loss_items += (loss.item()) * len(batch_inputs) # Accumulate loss items for logging
                    if lr_scheduler is not None:
                        lr_scheduler.step()

            epoch_end_time = time.time()
            avg_epoch_loss = total_epoch_loss_items / len(tensordata_loader.dataset) if len(tensordata_loader.dataset) > 0 else 0.0
            logging.info(f'Layer {i}, Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_epoch_loss}, Time: {epoch_end_time - epoch_start_time:.2f}s')

        # Final projection after optimization
        opt.final_projection()
    
        # Calculate outputs with the now-pruned layer
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0) # Inputs are already on device
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    current_layer_outputs[j] = layer(sample_input, attention_mask=attention_mask)[0].to('cpu')
                else:
                    current_layer_outputs[j] = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')

        layers[i] = layer.to('cpu') # Offload layer
        if args.activation:
            del importance_matrix
        del prunable_submodules,opt, lr_scheduler, tensordata, tensordata_loader
        torch.cuda.empty_cache()
        
        inps, current_layer_outputs = current_layer_outputs, inps # Swap for the next layer (outputs are on CPU)

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info("SAFE pruning finished.")

@torch.no_grad()
def prune_admm():
    pass

@torch.no_grad()
def prune_alps(args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using ALPS method.
    Original code: https://github.com/mazumder-lab/ALPS

    Args:
        args: Configuration object with sparsity_ratio (int), nsamples (int), seed (int), dataset (str), rho (float) (for ALPS_admm).
        model: The model to prune.
        tokenizer: The tokenizer for loading calibration data.
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info('Starting ALPS pruning...')
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.nsamples)

    layers = get_model_layers(model)

    current_layer_outputs = torch.zeros_like(inps)
    seqlen = model.seqlen 

    logging.info('ALPS calibration data prepared.')

    total_params_count = 0
    total_nnz_count = 0
    overall_pruning_start_time = time.time()

    # Pruning based on ALPS
    for i in range(len(layers)):
        layer_pruning_start_time = time.time()
        layer = layers[i].to(device) 
        current_inps = inps.to(device) 
        
        prunable_modules_map = find_layers(layer) 
        alps_pruner_instances = {} 
        logging.info(f'--- Pruning Layer {i} with ALPS ---')

        for name, module_to_prune in prunable_modules_map.items():
            alps_pruner_instances[name] = ALPS_prune(module_to_prune, nsamples=args.nsamples, seqlen=seqlen)

        def get_alps_add_batch_hook(module_name):
            def hook_fn(_, inp_tuple, out_tensor):
                actual_input_tensor = inp_tuple[0] if isinstance(inp_tuple, tuple) else inp_tuple
                alps_pruner_instances[module_name].add_batch(actual_input_tensor.data, out_tensor.data)
            return hook_fn

        handles = []
        for name in prunable_modules_map:
            handles.append(prunable_modules_map[name].register_forward_hook(get_alps_add_batch_hook(name)))
        
        # Pass calibration data through the layer to populate ALPS_prune buffers
        for j in range(args.nsamples):
            sample_input = current_inps[j].unsqueeze(0)
            if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                _ = layer(sample_input, attention_mask=attention_mask)[0] # Output captured by hooks
            else:
                _ = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0] # Output captured by hooks
        
        for h in handles: # Remove hooks
            h.remove()

        # Perform ALPS pruning for each module
        for name in prunable_modules_map:
            logging.info(f'Applying ALPS to layer {i}, module {name}')
            alps_pruner_instances[name].ALPS_admm(sp=args.sparsity_ratio, nm_n=prune_n, nm_m=prune_m)
            
            weight_data = alps_pruner_instances[name].layer.weight.data
            total_params_count += weight_data.numel()
            total_nnz_count += torch.count_nonzero(weight_data).item()
            
            alps_pruner_instances[name].free() 

        # Get outputs from the now-pruned layer
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    current_layer_outputs[j] = layer(sample_input, attention_mask=attention_mask)[0].to('cpu')
                else:
                    current_layer_outputs[j] = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')

        layers[i] = layer.to('cpu') 
        layer_pruning_end_time = time.time()
        logging.info(f'Layer {i} ALPS pruning time: {layer_pruning_end_time - layer_pruning_start_time:.2f}s')
        
        del layer, alps_pruner_instances, current_inps 
        torch.cuda.empty_cache()

        inps, current_layer_outputs = current_layer_outputs, inps 

    overall_pruning_end_time = time.time()
    logging.info(f'Total ALPS pruning time: {overall_pruning_end_time - overall_pruning_start_time:.2f}s')
    if total_params_count > 0:
        final_sparsity = 1 - (total_nnz_count / total_params_count)
        logging.info(f'Overall sparsity after ALPS: {final_sparsity:.4f} (NNZ: {total_nnz_count}, Total: {total_params_count})')

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info("ALPS pruning finished.")

@torch.no_grad()
def prune_sparsegpt(
    args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using the SparseGPT method.
    Original code: https://github.com/IST-DASLab/sparsegpt

    Args:
        args: Configuration object with sparsity_ratio (int), nsamples (int), seed (int), dataset (str).
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer for loading calibration data.
        device (torch.device): The default device. Layer-specific devices handled if model.hf_device_map exists.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info('Starting SparseGPT pruning...')
    
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.nsamples)

    layers = get_model_layers(model)

    current_layer_outputs = torch.zeros_like(inps) 
    logging.info('SparseGPT calibration data prepared.')

    # Pruning based on SparseGPT
    for i in range(len(layers)):
        layer = layers[i]
        current_layer_processing_device = device 

        layer_key_in_map = f"model.layers.{i}" 
        if hasattr(model, 'hf_device_map') and model.hf_device_map and layer_key_in_map in model.hf_device_map:
            current_layer_processing_device = model.hf_device_map[layer_key_in_map]
            logging.info(f"Layer {i} assigned to device: {current_layer_processing_device} from hf_device_map.")
        
        layer = layer.to(current_layer_processing_device)
        current_inps = inps.to(current_layer_processing_device)
        current_attention_mask = attention_mask.to(current_layer_processing_device) if attention_mask is not None else None
        current_position_ids = position_ids.to(current_layer_processing_device) if position_ids is not None else None


        prunable_submodules = find_layers(layer)
        sparsegpt_pruners = {}
        for name in prunable_submodules:
            sparsegpt_pruners[name] = SparseGPT(prunable_submodules[name])

        def get_sparsegpt_add_batch_hook(module_name):
            def hook_fn(_, inp_tuple, out_tensor):
                actual_input_tensor = inp_tuple[0] if isinstance(inp_tuple, tuple) else inp_tuple
                sparsegpt_pruners[module_name].add_batch(actual_input_tensor.data, out_tensor.data)
            return hook_fn

        handles = []
        for name in sparsegpt_pruners:
            handles.append(prunable_submodules[name].register_forward_hook(get_sparsegpt_add_batch_hook(name)))

        for j in range(args.nsamples):
            sample_input = current_inps[j].unsqueeze(0)
            if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                _ = layer(sample_input, attention_mask=current_attention_mask)[0] 
            else:
                _ = layer(sample_input, attention_mask=current_attention_mask, position_ids=current_position_ids)[0] 
        
        for h in handles: 
            h.remove()
            
        for name in sparsegpt_pruners:
            logging.info(f'Pruning layer {i}, module {name} with SparseGPT.')
            percdamp = getattr(args, 'sparsegpt_percdamp', 0.01)
            blocksize = getattr(args, 'sparsegpt_blocksize', 128)
            sparsegpt_pruners[name].fasterprune(
                args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, 
                percdamp=percdamp, blocksize=blocksize
            )
            sparsegpt_pruners[name].free()

        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    current_layer_outputs[j] = layer(sample_input, attention_mask=current_attention_mask)[0].to('cpu')
                else:
                    current_layer_outputs[j] = layer(sample_input, attention_mask=current_attention_mask, position_ids=current_position_ids)[0].to('cpu')

        layers[i] = layer.to('cpu') 
        del current_inps, current_attention_mask, current_position_ids, layer, sparsegpt_pruners
        torch.cuda.empty_cache()

        inps, current_layer_outputs = current_layer_outputs, inps

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info("SparseGPT pruning finished.")

### Global pruning

# --- Define AdmmTrainingArguments ---
@dataclass
class AdmmTrainingArguments(TrainingArguments):
    """
    Training arguments specific for ADMM training, inheriting from standard TrainingArguments.
    """
    # Add ADMM specific arguments that ADMMTrainer might need access to via the args object.
    # Standard TrainingArguments already cover most needs (lr, epochs, batch_size, etc.)
    # These might be redundant if ADMMTrainer reads them directly from FLAGS, but including
    # them here follows the pattern of passing config through the args object.
    admm_lmda: float = field(default=0.001, metadata={"help": "Lambda (rho) penalty parameter for ADMM."})
    admm_initial_lmda: float = field(default=0.0, metadata={"help": "Initial lambda (rho) for ADMM for penalty scheduling. defaults to 0.0."})
    admm_lmda_schedule_mode: str = field(default='constant', metadata={"help": "Mode for lambda schedule (linear/cosine/exponential/constant)."})
    admm_sparsity_schedule_mode: str = field(default='constant', metadata={"help": "Mode for sparsity schedule (linear/cosine/exponential/constant)."})
    admm_interval: int = field(default=32, metadata={"help": "Interval for ADMM projection and dual updates."})
    admm_projection_comparison_group: str = field(default='layer', metadata={"help": "Comparison group for ADMM projection (layer/column/row)."})
    prune_n: int = field(default=0, metadata={"help": "N for N:M sparsity."})
    prune_m: int = field(default=0, metadata={"help": "M for N:M sparsity."})
    sparsity_ratio: float = field(default=0.0, metadata={"help": "Target sparsity ratio (for reference)."})
    admm_peak_sparsity_step: float = field(default=1.0, metadata={"help": "Step at which peak sparsity is reached (for sparsity scheduling)."})
    base_optimizer_type: str = field(default='adam', metadata={"help": "Base optimizer for ADMM primal update."})
    blockwise_projection: bool = field(default=False, metadata={"help": "Use blockwise projection in ADMM."})
    activation_aware: bool = field(default=False, metadata={"help": "Use activation-aware projection in ADMM."})
    decouple_admm: bool = field(default=False, metadata={"help": "Decouple proximal update in ADMM (for Adam)."})
    is_safe: bool = field(default=False, metadata={"help": "Use SAFE pruning method."})
    rho: float = field(default=0.01, metadata={"help": "Rho parameter for SAFE pruning."})
    ## LOSS
    loss_type: str = field(default="ntp", metadata={"help": "Loss type for ADMM training (should be 'rem' or 'ntp)."})

# --- globalprune_admm function ---
def globalprune_admm(FLAGS, model, tokenizer, device, prune_n=0, prune_m=0):
    """
    Performs ADMM training globally.
    """
    if FLAGS.admm_save_path:
        model_name_part = FLAGS.model.split('/')[-1]
        admm_run_name = f"{model_name_part}_pruned{FLAGS.sparsity_ratio}_admm_lr{FLAGS.admm_lr}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        admm_output_dir = Path(FLAGS.admm_save_path) / admm_run_name
        admm_output_dir.mkdir(parents=True, exist_ok=True)
        admm_output_dir_str = str(admm_output_dir)
    else:
        admm_output_dir_str = f"./admm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    admm_training_args = AdmmTrainingArguments(
        output_dir=admm_output_dir_str,
        num_train_epochs=FLAGS.admm_epochs,
        max_steps=FLAGS.admm_steps if FLAGS.admm_steps > 0 else -1,
        per_device_train_batch_size=FLAGS.admm_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=FLAGS.admm_gradient_accumulation_steps,
        learning_rate=FLAGS.admm_lr,
        lr_scheduler_type=FLAGS.admm_lr_scheduler,
        warmup_steps=FLAGS.admm_warmup_steps,
        weight_decay=FLAGS.admm_weight_decay,
        gradient_checkpointing=FLAGS.admm_gradient_checkpointing,
        fp16=(FLAGS.admm_precision == 'fp16'),
        bf16=(FLAGS.admm_precision == 'bf16' and torch.cuda.is_bf16_supported()),
        logging_steps=FLAGS.admm_logging_steps,
        evaluation_strategy="steps",
        logging_strategy="steps",
        eval_steps=FLAGS.admm_eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_ce_loss",
        greater_is_better=False,
        report_to="wandb" if has_wandb and FLAGS.wandb else "none",
        remove_unused_columns=False,
        do_train=True,
        do_eval=True,
        # admm arguments
        admm_alpha=FLAGS.admm_alpha,
        admm_lmda=FLAGS.admm_lmda,
        admm_initial_lmda=FLAGS.admm_initial_lmda,
        admm_lmda_schedule_mode=FLAGS.admm_lmda_schedule_mode,
        sparsity_ratio=FLAGS.sparsity_ratio,
        admm_peak_sparsity_step=FLAGS.admm_peak_sparsity_step,
        admm_sparsity_schedule_mode=FLAGS.admm_sparsity_schedule_mode,
        admm_interval=FLAGS.admm_interval,
        base_optimizer_type=FLAGS.admm_base_optimizer,
        admm_projection_comparison_group=FLAGS.admm_projection_comparison_group,
        prune_n=prune_n,
        prune_m=prune_m,
        loss_type=FLAGS.loss_type,
        #safe
        is_safe=FLAGS.is_safe,  # Use this flag to determine if SAFE pruning is used
        rho=FLAGS.rho,  # Rho parameter for SAFE pruning
    )

    # --- 로깅은 메인 프로세스에서만 수행 ---
    if admm_training_args.local_rank == 0:
        logging.info("--- Starting Global Pruning via ADMM Training  ---")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        if admm_training_args.local_rank == 0:
            logging.info(f"Tokenizer pad_token not set. Using eos_token as pad_token.")

    # 1. Prepare Datasets for ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Preparing dataset for ADMM training...")
    # Use ADMM specific flags for dataset parameters
    if FLAGS.admm_steps > 0:
        num_train_samples = FLAGS.admm_steps * FLAGS.admm_batch_size * FLAGS.admm_gradient_accumulation_steps
    else:
        num_train_samples = FLAGS.admm_num_train_samples
    # Ensure model's seqlen matches the one used for dataset processing
    model.seqlen = FLAGS.seqlen
    train_inputs = get_dataset(
        dataset_name=FLAGS.dataset,
        tokenizer=tokenizer,
        nsamples=num_train_samples,
        seed=FLAGS.seed,
        seqlen=model.seqlen,
        data_type="train",
        save_to_cache=FLAGS.admm_save_inputs
    )
    valid_inputs = get_dataset(
        dataset_name=FLAGS.dataset,
        tokenizer=tokenizer,
        nsamples=FLAGS.admm_num_eval_samples,
        seed=FLAGS.seed,
        seqlen=model.seqlen,
        data_type="validation", # Use a different data_type for eval
        save_to_cache=FLAGS.admm_save_inputs
    )

    # --- On-the-fly REM Label Computation for Distributed Training ---
    if FLAGS.loss_type == "rem":
        # Tensors to hold the labels, initialized to None on all processes
        objects_to_broadcast = [None, None]

        # Rank 0 computes the labels for the entire dataset
        if admm_training_args.local_rank == 0:
            logging.info("Rank 0: Computing dense model outputs for REM loss...")
            torch.cuda.empty_cache()

            # Compute labels and move them to CPU for broadcasting
            train_labels_tensor = compute_dense_outputs(train_inputs, model, FLAGS, device=device, dataset_type="train").to("cpu")
            valid_labels_tensor = compute_dense_outputs(valid_inputs, model, FLAGS, device=device, dataset_type="valid").to("cpu")
            
            objects_to_broadcast = [train_labels_tensor, valid_labels_tensor]
            logging.info("Rank 0: Finished computing REM labels. Broadcasting to other processes...")

        # Synchronize all processes and broadcast the computed labels from rank 0
        if admm_training_args.world_size > 1:
            torch.distributed.barrier()  # Ensure rank 0 is done before others proceed
            torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)

        # Unpack the broadcasted labels on all processes
        train_labels_tensor, valid_labels_tensor = objects_to_broadcast

        # Add the computed labels to the datasets
        if train_labels_tensor is not None and valid_labels_tensor is not None:
            train_labels_dataset = Dataset.from_dict({'rem_labels': train_labels_tensor.numpy()})
            valid_labels_dataset = Dataset.from_dict({'rem_labels': valid_labels_tensor.numpy()})

            train_inputs = concatenate_datasets([train_inputs, train_labels_dataset], axis=1)
            valid_inputs = concatenate_datasets([valid_inputs, valid_labels_dataset], axis=1)
            
            if admm_training_args.local_rank == 0:
                logging.info("REM labels computed and added to datasets on all processes.")
    
    if admm_training_args.local_rank == 0:
        logging.info(f"ADMM Datasets prepared: Train size {len(train_inputs)}, Valid size {len(valid_inputs)}")

    model.train()
    # 4. Initialize ADMMTrainer
    if admm_training_args.local_rank == 0:
        logging.info("Initializing ADMMTrainer...")
    trainer = ADMMTrainer(
        model=model, # Pass the model while it's on the CPU
        args=admm_training_args,
        train_dataset=train_inputs,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # 5. Start ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Starting ADMM training on all processes...")
    
    trainer.train()
