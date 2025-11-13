from absl import logging
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import fnmatch
from .data import get_loaders 

# Code adopted from https://github.com/locuslab/wanda

def eval_ppl(
    args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device = torch.device("cuda:0"),
    data_path: str = None
) -> dict:
    """
    Evaluate the model on the wikitext2 and c4 datasets.
    Args:
        args: Namespace, command line arguments.
        model (AutoModelForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        device (torch.device): The device to use for evaluation.
        data_path: The path to the dataset.
    Returns:
        dict: A dictionary containing the perplexity (ppl) for each dataset.
    """
    dataset = ["wikitext2", "c4"]
    ppls = defaultdict(float)
    for d in dataset:
        # Print status
        logging.info(f"evaluating on {d}")

        # Get the test loader
        _, testloader = get_loaders(
            d, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer, data_path=data_path 
        )
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = calculate_ppl(model, testloader,tokenizer, 1)
            ppls[d] = ppl_test
    return ppls 

# @torch.no_grad()
# def calculate_ppl(
#     model: AutoModelForCausalLM,
#     testenc,
#     tokenizer: AutoTokenizer,
#     bs: int = 1
# ) -> float:

#     seqlen = model.seqlen
#     testenc = testenc.input_ids
#     nsamples = testenc.numel() // seqlen

#     nlls = []
    
#     for i in range(0, nsamples, bs):
#         j = min(i + bs, nsamples)
#         batch_size = j - i

#         # Special handling for Gemma model architecture
#         is_gemma = 'Gemma' in model.config.architectures[0]
        
#         if is_gemma:
#             # 1. Slice the data to seqlen - 1 to make space for BOS token.
#             inputs_slice = testenc[:, (i * seqlen):(i * seqlen + (batch_size * (seqlen - 1)))].to(model.device)
#             inputs_slice = inputs_slice.reshape(batch_size, seqlen - 1)
            
#             # 2. Prepend BOS token to match the final length to seqlen.
#             bos_tensor = torch.tensor([[tokenizer.bos_token_id] * batch_size]).reshape(batch_size, 1).to(model.device)
#             model_inputs = torch.cat([bos_tensor, inputs_slice], dim=1)

#             # 3. Labels (ground truth) are the original text sequence excluding BOS.
#             shift_labels = inputs_slice.contiguous()

#         else: # For models other than Gemma
#             model_inputs = testenc[:, (i * seqlen):(j * seqlen)].to(model.device)
#             model_inputs = model_inputs.reshape(batch_size, seqlen)
#             shift_labels = model_inputs[:, 1:].contiguous()

#         # Model forward pass
#         with torch.no_grad():
#             lm_logits = model(model_inputs).logits

#         # Reorder logits for loss calculation
#         # Gemma: L_bos, L_t1, ... L_t(n-2) => total n-1 predictions, matched with labels T1, T2 ... T(n-1)
#         # Other: L_t1, L_t2, ... L_t(n-1) => total n-1 predictions, matched with labels T2, T3 ... Tn
#         shift_logits = lm_logits[:, :-1, :].contiguous()

#         loss_fct = nn.CrossEntropyLoss()
#         # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32), shift_labels.view(-1))

#         neg_log_likelihood = loss.float() * (shift_labels.numel() / batch_size) # Calculate NLL based on actual label length
#         nlls.append(neg_log_likelihood)

#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
#     torch.cuda.empty_cache()

#     return ppl.item()




@torch.no_grad()
def calculate_ppl(
    model: AutoModelForCausalLM,
    testenc,
    tokenizer: AutoTokenizer,
    bs: int = 1
) -> float:

    seqlen = model.seqlen
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    nlls = []
    
    for i in range(0, nsamples, bs):
        j = min(i + bs, nsamples)
        batch_size = j - i

        # First cut the input to seqlen length for all models
        model_inputs = testenc[:, (i * seqlen):(j * seqlen)].to(model.device)
        model_inputs = model_inputs.reshape(batch_size, seqlen)

        # Special handling for Gemma model architecture
        is_gemma = 'Gemma' in model.config.architectures[0]
        
        # === Modified part: Gemma handling logic ===
        if is_gemma:
            # 1. Replace the first token of existing sequence with BOS token
            model_inputs[:, 0] = tokenizer.bos_token_id

        # 2. Label generation logic applies equally to both Gemma and other models
        # [BOS, t2, t3, ...] -> labels: [t2, t3, ...]
        # [t1, t2, t3, ...] -> labels: [t2, t3, ...]
        shift_labels = model_inputs[:, 1:].contiguous()
        # ==================================

        # Model forward pass
        with torch.no_grad():
            lm_logits = model(model_inputs).logits

        # Reorder logits for loss calculation
        # Excluding the last logit makes the length equal to prediction targets (labels)
        shift_logits = lm_logits[:, :-1, :].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * (shift_labels.numel() / batch_size) # Calculate NLL based on actual label length
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    torch.cuda.empty_cache()

    return ppl.item()




def eval_zero_shot(args,model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    tm = tasks.TaskManager()
    task_names = pattern_match(task_list, tm.all_tasks)
    # model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None # for testing purpose. use None to evaluate on all examples 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    from lm_eval.models.huggingface import HFLM
    if "70b" in model_name or "65b" in model_name:
        model = HFLM(model,parallelize=True,max_memory_per_gpu="40GB")
    else:
        model = HFLM(model)
    results = evaluator.simple_evaluate(
        model=model,
        # model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        cache_requests=False,
        batch_size="auto",
        device=model.device,
        use_cache=None,
        limit=limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        check_integrity=False
    )
    results = results['results'] ## return only the results
    return results 
