import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, EvalPrediction
import math
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor, Shard
import torch.distributed as dist
from typing import Optional, List
def get_llm(
    model_name:str, 
    seqlen:int=2048
)-> AutoModelForCausalLM:
    """
    Load the model from huggingface hub or local directory.
    The model should be a causal language model, such as Llama2, Gemma, etc.
    Args:
        Model_name: str, directly from huggingface hub, or the directory of the model.
        seqlen: int, the maximum sequence length for the model.
    Returns:
        model: AutoModelForCausalLM, the model loaded from huggingface hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code = True,
        use_flash_attention_2=False
    )
    assert seqlen<=model.config.max_position_embeddings, f"seqlen({seqlen}) should be less than or equal to model.config.max_position_embeddings({model.config.max_position_embeddings})"
    model.seqlen = seqlen
    return model

def _as_dense_a(a):
    """Convert importance matrix to dense tensor if it is DTensor."""
    if a is None:
        return None
    if isinstance(a, DTensor):
        return a.redistribute(placements=[Replicate()]).to_local()
    return a






def _proj_impl_dense(
    weight: torch.Tensor,
    a: Optional[torch.Tensor],
    sparsity: float,
    prune_n: int,
    prune_m: int,
    comparison_group: str,
    is_direct_score: bool = False,
) -> torch.Tensor:
    """
    Projection core for dense tensors.
    - Works entirely on local tensor (no communication).
    - Supports unstructured and n:m semi-structured sparsity.
    - comparison_group:
        'layer'  : global threshold (within this tensor)
        'column' : prune k smallest entries per column
        'row'    : prune k smallest entries per row
    """
    device = weight.device
    new_z = weight.detach().clone()

    if a is not None:
        if is_direct_score:
            z_metric = a
        else:
            # Generalized projection: metric is V * x^2 where x = weight and a = V
            z_metric = a * (weight**2)
    else:
        # Standard projection: metric is |x|
        z_metric = weight.abs()

    if prune_n != 0 and prune_m != 0:
        # n:m semi-structured pruning (column-block based)
        z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
        cols = z_metric.shape[1]
        for ii in range(0, cols, prune_m):
            blk = z_metric[:, ii:ii + prune_m].float()
            if blk.numel() == 0:
                continue
            k = min(prune_n, blk.shape[1])
            if k <= 0:
                continue
            # Select k smallest entries per row in the block
            _, idxs = torch.topk(blk, k=k, dim=1, largest=False)
            z_mask.scatter_(1, ii + idxs, True)
        new_z[z_mask] = 0
        return new_z

    # ---- Unstructured sparsity ----
    if comparison_group == "layer":
        # Global threshold within this tensor
        k = int(new_z.numel() * sparsity)
        if k > 0:
            flat_sorted = torch.sort(z_metric.flatten(), stable=True)[0]
            kth = flat_sorted[min(k - 1, flat_sorted.numel() - 1)]
            new_z[z_metric <= kth] = 0
        return new_z

    elif comparison_group == "column":
        # Prune per column: smallest k rows
        num_rows_to_prune_per_col = int(new_z.shape[0] * sparsity)
        if num_rows_to_prune_per_col > 0:
            z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
            _, idx = torch.topk(z_metric, k=num_rows_to_prune_per_col, dim=0, largest=False)
            z_mask.scatter_(0, idx, True)
            new_z[z_mask] = 0
        return new_z

    else:  # 'row'
        # Prune per row: smallest k columns
        num_cols_to_prune_per_row = int(new_z.shape[1] * sparsity)
        if num_cols_to_prune_per_row > 0:
            z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
            _, idx = torch.topk(z_metric, k=num_cols_to_prune_per_row, dim=1, largest=False)
            z_mask.scatter_(1, idx, True)
            new_z[z_mask] = 0
        return new_z

def projection(
    w: List[torch.Tensor],
    sparsity: float,
    prune_n: int = 0,
    prune_m: int = 0,
    importance_matrix: Optional[List[torch.Tensor]] = None,
    comparison_group: str = "layer",
    is_direct_score: bool = False,
) -> List[torch.Tensor]:
    """
    Distributed/DTensor-friendly projection.
    If weight is a DTensor, it is always replicated before projection.
    """
    assert comparison_group in ("layer", "column", "row")
    use_a = importance_matrix is not None
    if use_a:
        assert len(importance_matrix) == len(w)

    out: List[torch.Tensor] = []
    for i, weight in enumerate(w):
        a = importance_matrix[i] if use_a else None

        if isinstance(weight, DTensor):
            # Always replicate DTensor to a dense tensor for projection
            mesh = weight.device_mesh
            orig_places = weight.placements
            
            dense_w = _as_dense_a(weight)
            dense_a = _as_dense_a(a)

            new_dense = _proj_impl_dense(dense_w, dense_a, sparsity, prune_n, prune_m, comparison_group, is_direct_score)
            
            # Redistribute the result back to the original sharding
            new_dt = distribute_tensor(new_dense, device_mesh=mesh, placements=orig_places)
            out.append(new_dt)
        else:
            # It's a regular dense tensor
            new_dense = _proj_impl_dense(weight, _as_dense_a(a), sparsity, prune_n, prune_m, comparison_group, is_direct_score)
            out.append(new_dense)
            
    return out

# def projection(
#     w: list[torch.Tensor],
#     sparsity: float,
#     prune_n: int =0,
#     prune_m: int = 0,
#     importance_matrix: list[torch.Tensor] = None,
#     comparison_group: str = 'layer'
# ) -> list[torch.Tensor]:
#     """
#     Args:
#         w (list[torch.Tensor]): list of weights (nxm) to be projected
#         sparsity (float): target sparsity
#         prune_n (int): n for n:m semi-structured sparsity
#         prune_m (int): m for n:m semi-structured sparsity
#         importance_matrix (list[torch.Tensor], optional): importance matrix (diag(mxm)) or vector (1xm) for each weight
#         comparison_group (str): 'layer' for layer-wise pruning metric comparison, 'column' for column-wise(input), 'row' for row-wise(output) pruning metric comparison.
#     Returns:
#         new_zs (list[torch.Tensor]): list of projected weights
#     """
#     new_zs = []
#     if importance_matrix is not None: # Generalized projection, SAFE+
#         for weight,a in zip(w,importance_matrix):
#             new_z = weight.data.clone().detach()
#             z_metric = torch.abs(weight) * a
#             if prune_n != 0: # n:m semi-structured sparsity
#                 z_mask = (torch.zeros_like(new_z)==1)
#                 for ii in range(z_metric.shape[1]):
#                     if ii % prune_m == 0:
#                         tmp = z_metric[:,ii:(ii+prune_m)].float()
#                         z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
#             else: # unstructured sparsity
#                 z_mask = (torch.zeros_like(new_z)==1)
#                 if comparison_group == 'layer':
#                     thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()                
#                     z_mask = (z_metric<=thresh)
#                 elif comparison_group == 'column':
#                     num_rows_to_prune_per_col = int(new_z.shape[0]*sparsity)
#                     if num_rows_to_prune_per_col > 0:
#                         _, indices_to_prune = torch.topk(
#                             z_metric,
#                             k=num_rows_to_prune_per_col,
#                             dim=0,
#                             largest=False
#                         )
#                     z_mask.scatter_(0,indices_to_prune, True)
#                 elif comparison_group == 'row':
#                     num_cols_to_prune_per_row = int(new_z.shape[1]*sparsity)
#                     if num_cols_to_prune_per_row > 0:
#                         _, indices_to_prune = torch.topk(
#                             z_metric,
#                             k=num_cols_to_prune_per_row,
#                             dim=1,
#                             largest=False
#                         )
#                     z_mask.scatter_(1,indices_to_prune, True)
#             new_z[z_mask] = 0
            
#             new_zs.append(new_z)
#     else: # Standard projection, SAFE
#         for weight in w:
#             new_z = weight.data.clone().detach()
#             z_metric = torch.abs(weight)
#             if prune_n != 0: # n:m semi-structured sparsity
#                 z_mask = (torch.zeros_like(new_z)==1)
#                 for ii in range(z_metric.shape[1]):
#                     if ii % prune_m == 0:
#                         tmp = z_metric[:,ii:(ii+prune_m)].float()
#                         z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
#             else: # unstructured sparsity
#                 z_mask = (torch.zeros_like(new_z)==1)
#                 if comparison_group == 'layer':
#                     thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()                
#                     z_mask = (z_metric<=thresh)
#                 elif comparison_group == 'column':
#                     num_rows_to_prune_per_col = int(new_z.shape[1]*sparsity)
#                     if num_rows_to_prune_per_col > 0:
#                         _, indices_to_prune = torch.topk(
#                             z_metric,
#                             k=num_rows_to_prune_per_col,
#                             dim=1,
#                             largest=False
#                         )
#                     z_mask.scatter_(1,indices_to_prune, True)
#                 elif comparison_group == 'row':
#                     num_cols_to_prune_per_row = int(new_z.shape[0]*sparsity)
#                     if num_cols_to_prune_per_row > 0:
#                         _, indices_to_prune = torch.topk(
#                             z_metric,
#                             k=num_cols_to_prune_per_row,
#                             dim=0,
#                             largest=False
#                         )
#                     z_mask.scatter_(0,indices_to_prune, True)
#             new_z[z_mask] = 0
#             new_zs.append(new_z)
#     return new_zs
 
def find_layers(
    module: nn.Module,
    layers: list = [nn.Linear],
    name: str = ''
) -> dict:
    """
    Recursively find the layers of a certain type in a module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find. 
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_model_layers(model):
    """
    Returns the list of Transformer layers based on the model architecture.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.ModuleList: The list of Transformer layers in the model.

    Raises:
        ValueError: If the model architecture is unsupported.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama, Gemma, Mistral, etc.
        return model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT, etc.
        return model.model.decoder.layers
    else:
        raise ValueError("Unsupported model architecture: Cannot find layers.")

def get_model_embeddings(model):
    """
    Returns the input embedding layer based on the model architecture.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.Module: The input embedding layer of the model.

    Raises:
        ValueError: If the model architecture is unsupported.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'embed_tokens'):
        return model.model.decoder.embed_tokens
    else:
        raise ValueError("Unsupported model architecture: Cannot find embedding layer.")

def get_model_norm(model):
    """
    Returns the final Layer Normalization layer based on the model architecture, if it exists.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.Module or None: The final Layer Normalization layer of the model, or None.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        # Llama, Gemma, etc.
        return model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'final_layer_norm'):
        # OPT, etc.
        return model.model.decoder.final_layer_norm
    else:
        # Other models might not have a final norm
        return None

def get_model_rotary_emb(model):
    """
    Returns the Rotary Embedding module based on the model architecture, if it exists.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.Module or None: The Rotary Embedding module of the model, or None.
    """
    # Currently assumes Llama-style models primarily
    if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
        return model.model.rotary_emb
    else:
        return None

def check_sparsity(model, log_by_block: bool = False):
    """
    Calculates the sparsity (ratio of zero parameters) of the model's linear layers.

    Args:
        model (nn.Module): The model object to check sparsity for.
        log_by_block (bool): If True, logs the sparsity for each transformer block.

    Returns:
        float: The overall sparsity ratio of the linear layers in the model (0.0 to 1.0).
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_model_layers(model)

    count = 0
    total_params = 0
    
    if log_by_block:
        logging.info("Checking sparsity per block...")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer) # Find linear layers within the layer

        sub_count = 0
        sub_params = 0
        for name in subset:
            # Check if the layer has a weight parameter
            if hasattr(subset[name], 'weight') and subset[name].weight is not None:
                W = subset[name].weight.data
                
                zeros = (W == 0).sum().item()
                total = W.numel()

                count += zeros
                total_params += total
                
                sub_count += zeros
                sub_params += total

        if log_by_block:
            layer_sparsity = float(sub_count) / sub_params if sub_params > 0 else 0.0
            logging.info(f"  - Block {i:02d} sparsity: {layer_sparsity:.4f}")

    model.config.use_cache = use_cache
    overall_sparsity = float(count) / total_params if total_params > 0 else 0.0
    
    if log_by_block:
        logging.info(f"Overall linear layer sparsity: {overall_sparsity:.4f}")
        
    return overall_sparsity

def prepare_calibration_input(
    model:AutoModelForCausalLM,
    dataloader:torch.utils.data.DataLoader,
    device:torch.device,
    nsamples:int=128
)-> tuple:
    """
    Prepare input data for model calibration.
    Supports OpenLM models and HF models (Llama2, Llama3, Gemma2).
    Offloads most of the model to CPU, loading only necessary parts to the device to maximize memory efficiency.
    Captures the activations for the first transformer layer's input.
    Args:
        model (AutoModelForCausalLM): The model for which to prepare calibration data.
        dataloader (torch.utils.data.DataLoader): DataLoader providing calibration data.
        device (torch.device): The device to use for capturing activations.
        nsamples (int): Number of samples to prepare.

    Returns:
        tuple: (inps, outs, attention_mask, position_ids)
            inps (torch.Tensor): Input activations to the first transformer block.
            outs (torch.Tensor): Placeholder for outputs (same shape as inps).
            attention_mask (torch.Tensor): Attention mask from calibration data.
            position_ids (torch.Tensor): Position IDs from calibration data.
    """
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    # Use helper functions to robustly get model components
    layers = get_model_layers(model)
    embed_tokens = get_model_embeddings(model)
    norm = get_model_norm(model)
    rotary_emb = get_model_rotary_emb(model)

    # Move necessary components to the specified device
    embed_tokens.to(device)
    if norm is not None:
        norm.to(device)
    if rotary_emb is not None:
        rotary_emb.to(device)
        if hasattr(rotary_emb, 'inv_freq'):
            rotary_emb.inv_freq = rotary_emb.inv_freq.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(model.config, 'dim', None)
        if hidden_size is None:
            raise ValueError("Could not find hidden_size or dim in model config")
    
    inps = torch.zeros((nsamples, model.seqlen, hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Helper module to catch inputs to the first transformer layer
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            # Handle cases where the input is a tuple (e.g., in OPT models)
            input_tensor = inp[0] if isinstance(inp, tuple) else inp
            if cache['i'] < nsamples: 
                 inps[cache['i']] = input_tensor.detach() 
            cache['i'] += 1
            if 'attention_mask' in kwargs:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs: 
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError 
    
    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0]) 
    
    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError: # Expected exception from Catcher
            pass 
        samples_collected = min(cache['i'], nsamples)


    layers[0] = original_first_layer # Restore original layer
    
    # Offload parts from device
    # layers[0] = layers[0].to('cpu')
    # embed_tokens.to('cpu')
    # if norm is not None:
    #     norm.to('cpu')
    # if rotary_emb is not None:
    #     rotary_emb.to('cpu')

    # Finalize outputs
    # If fewer than nsamples were collected, slice inps
    if samples_collected < nsamples:
        logging.warning(f"Collected {samples_collected} samples, less than requested {nsamples}.")
        inps = inps[:samples_collected]
    
    inps = inps.to('cpu') # Move collected inputs to CPU
    outs = torch.zeros_like(inps) # Placeholder for outputs
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return inps, outs, attention_mask, position_ids


## REM utils
def calculate_reconstruction_error(
    inps:torch.Tensor,
    outs:torch.Tensor,
    device:torch.device
)-> float:
    """
    Calculates the mean squared error between inputs and outputs in FP32.

    Args:
        inps (torch.Tensor): Input tensors.
        outs (torch.Tensor): Output tensors.
        device (torch.device): Device to use for calculation (though result is CPU).

    Returns:
        float: The reconstruction error (MSE) on CPU.
    
    Note:
        MSE calculation on GPU can be faster, but this function returns a CPU scalar.
        It's typically not a bottleneck if called infrequently (e.g., per block).
    """
    with torch.no_grad():
        inps_device = inps.to(device, non_blocking=True)
        outs_device = outs.to(device, non_blocking=True)
        return torch.nn.MSELoss()(inps_device.float(), outs_device.float()).cpu().item()
    
@torch.no_grad()
def compute_dense_outputs(dataset, model, args, device="cuda", dataset_type="train", batch_size=4):
    """
    Compute and store hidden states from the dense model using hooks.
    Args:
        dataset: Input dataset containing tokenized texts ids and attention mask
        model: Dense model to extract outputs from
        args: Training arguments
        device: Device to run the model on
        dataset_type: Type of dataset ('train' or 'validation')
    Returns:
        Tuple of (input_ids, hidden_states)
    """
    import gc
    import os
    from datetime import datetime
    import json
    from tqdm import tqdm
    # Setup output directory structure
    safe_model_name = args.model.replace("/", "_")
    base_dir = os.path.join("dataset", safe_model_name)
    dataset_dir = os.path.join(base_dir, "train" if dataset_type == "train" else "valid")
    os.makedirs(dataset_dir, exist_ok=True)
    # Define file paths
    input_file = os.path.join(dataset_dir, "inputs.pt")
    labels_file = os.path.join(dataset_dir, "labels.pt")
    metadata_file = os.path.join(dataset_dir, "labels_metadata.json")
    # Check if files exist and load metadata
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if (metadata.get('model_name') == args.model and
                metadata.get('dataset_type') == dataset_type and
                metadata.get('source_dataset') == args.dataset and
                metadata.get('num_samples') == (args.admm_num_train_samples if dataset_type == "train" else args.admm_num_eval_samples) and
                metadata.get('seqlen') == args.seqlen):
                print(f"Loading cached {dataset_type} dense outputs...")
                labels = torch.load(labels_file, weights_only=False)
                return labels
    print("No valid cached labels data found. Processing datasets from input data...")
    print(f"Computing dense outputs for {dataset_type} dataset...")
    # Prepare model
    model.eval()
    model = model.to(device)
    try:
        # Initialize lists to store results
        all_hidden_states = []
        # Process dataset in batches
        batch_size = 4  # Adjust based on GPU memory
        num_samples = len(dataset)
        print(f"Processing {num_samples} samples in batches of {batch_size}")
        for i in tqdm(range(0, num_samples, batch_size)):
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            # Get batch
            batch_end = min(i + batch_size, num_samples)
            batch = dataset[i:batch_end]
            # Move inputs to device
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            # # Forward pass
            # outputs = model(input_ids, attention_mask=attention_mask)
            # hidden_states = activations['output']
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            # Store results
            all_hidden_states.append(hidden_states.cpu())
            # Clean up
            del input_ids, attention_mask, outputs, hidden_states
            gc.collect()
        # Concatenate results
        all_hidden_states = torch.cat(all_hidden_states, dim=0)
        # Verify shapes
        expected_samples = args.admm_num_train_samples if dataset_type == "train" else args.admm_num_eval_samples
        assert len(dataset['input_ids']) == expected_samples, f"Expected {expected_samples} samples, got {len(dataset['input_ids'])}"
        assert len(dataset['input_ids'][0]) == args.seqlen, f"Expected sequence length {args.seqlen}, got {len(dataset['input_ids'][0])}"
        assert all_hidden_states.size(0) == expected_samples
        assert all_hidden_states.size(1) == args.seqlen
        # Save results
        print(f"Saving {dataset_type} dense outputs...")
        torch.save(all_hidden_states, labels_file)
        # Save metadata
        metadata = {
            'model_name': args.model,
            'dataset_type': dataset_type,
            'source_dataset': args.dataset,
            'num_samples': expected_samples,
            'seqlen': args.seqlen,
            'hidden_dim': all_hidden_states.size(-1),
            'created_at': datetime.now().isoformat(),
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        return all_hidden_states
    finally:
        # Clean up hook
        # hook_handle.remove()
        # Move model back to CPU
        model.to('cpu')
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

# gradient masking
def mask_grad(model):
    if hasattr(model.model, 'layers'): # for llama models
        layers = model.model.layers
    else: # for opt models
        layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            mask = (W==0)
            subset[name].weight.grad[mask]= 0
