import torch
from typing import Union, Dict
import torch.distributed as dist
from torch.distributed.tensor import DTensor
## TODO: ADD support for fsdp2 (projection - full tensor)

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
import math 
def _is_dtensor(x): 
    return hasattr(x, "to_local")

def _loc(x):
    # Return local shard if DTensor, otherwise the tensor itself
    return x.to_local() if _is_dtensor(x) else x

class ADMM(torch.optim.Adam):
    """
    ADMM optimizer built by subclassing AdamW (single optimizer object).
    - Proximal term is added AFTER gradient clipping and BEFORE the actual step.
    - Compatible with FSDP2/DTensor: all state kept per-shard, reductions in fp32.
    - Preserves original attributes: alpha, interval, sparsity, mask_diff, projection modes, EMA, etc.
    """
    def __init__(
        self,
        param_groups,
        projection_fn,
        sparsity: float,
        interval: int,
        alpha: float = 1.0,
        lmda: float = 1e-3, # For constant schedule
        init_lmda: float = 0.0, # For scheduling
        final_lmda: float = 0.01, # For scheduling
        lmda_schedule_mode: str = 'constant', # New: 'constant', 'linear', 'cosine', 'adaptive_boyd'
        total_steps: int = 1, # New: Total steps for fixed lmda schedules
        mu: float = 1.5, # For adaptive penalty
        tau_incr: float = 2.0, # For adaptive penalty
        tau_decr: float = 2.0, # For adaptive penalty
        prune_n: int = 0,
        prune_m: int = 0,
        comparison_group: str = "layer",
        projection_mode: str = "identity",   # 'identity' | 'gradient' | 'activation'
        importance_ema: float = 0.0,
        decouple: bool = False,
        dual_dtype: str = 'fp32',
        split_dtype: str = 'fp32',
        accelerator=None,                    # optional: to get world_size and device
        **adamw_kwargs
    ):
        super().__init__(param_groups, **adamw_kwargs)

        # --- ADMM config (mirrors your original) ---
        self.projection      = projection_fn
        self.sparsity        = float(sparsity)
        self.interval        = int(interval)
        self.alpha           = float(alpha)
        self.init_lmda = float(init_lmda)
        self.final_lmda = float(final_lmda)
        self.lmda_schedule_mode = lmda_schedule_mode.lower()

        if self.lmda_schedule_mode == 'constant':
            self.lmda_default = float(lmda)
        else:
            self.lmda_default = float(init_lmda) # Use init_lmda as the default starting value
        self.total_steps     = int(total_steps)
        self.mu              = float(mu)
        self.tau_incr        = float(tau_incr)
        self.tau_decr        = float(tau_decr)
        self.prune_n         = int(prune_n)
        self.prune_m         = int(prune_m)
        self.comparison_group = comparison_group.lower()
        self.projection_mode  = projection_mode.lower()
        self.importance_ema   = float(importance_ema)
        self.decouple       = bool(decouple)

        if self.lmda_schedule_mode != 'constant' and self.init_lmda is None:
            raise ValueError("For lambda scheduling, init_lmda must be provided.")

        if dual_dtype == 'bf16':
            self.dual_dtype = torch.bfloat16
        elif dual_dtype == 'fp32':
            self.dual_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dual_dtype: {dual_dtype}")

        if split_dtype == 'bf16':
            self.split_dtype = torch.bfloat16
        elif split_dtype == 'fp32':
            self.split_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported split_dtype: {split_dtype}")

        if self.comparison_group not in ("layer", "column", "row"):
            raise ValueError(f"comparison_group must be 'layer'|'column'|'row', got {self.comparison_group}")
        if self.projection_mode not in ("identity", "gradient", "activation", "momentum", "taylor"):
            raise ValueError(f"projection_mode must be 'identity'|'gradient'|'activation'|'momentum'|'taylor', got {self.projection_mode}")
        if self.lmda_schedule_mode not in ('constant', 'linear', 'cosine', 'exponential', 'adaptive_boyd'):
            raise ValueError(f"lmda_schedule_mode must be 'constant', 'linear', 'cosine', 'exponential', or 'adaptive_boyd', got {self.lmda_schedule_mode}")

        # Runtime helpers
        self.accelerator = accelerator
        self.current_step = 0
        self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0}

    def _lazy_init_admm_state(self, p: torch.nn.Parameter, group: Dict):
        """
        Lazily initialize all required states for a parameter for both ADMM and the base Adam optimizer.
        If ADMM state (`dual`) already exists, this function does nothing.
        This prevents conflicts with the base optimizer's own state initialization.
        """
        st = self.state[p]
        if 'dual' in st:
            return

        # --- Initialize Adam's state (if not already present) ---
        if 'exp_avg' not in st:
            st["step"] = torch.tensor(0.0)
            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group.get("amsgrad", False):
                st["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        # --- Initialize ADMM's state ---
        st["dual"] = torch.zeros_like(p, dtype=self.dual_dtype, memory_format=torch.preserve_format)
        st["sparsity"] = self.sparsity
        st["lmda"] = group.get("lmda", self.lmda_default)
        st["prev_lmda"] = st["lmda"]

        # Optional importance buffer
        init_importance = None
        if self.projection_mode in ("gradient", "activation"):
            st["importance"] = torch.zeros(_loc(p).shape[-1], device=p.device)
            init_importance = st["importance"]
        elif self.projection_mode == "momentum":
            init_importance = None

        # Initial split z and initial_split (as bool)
        z0 = self.projection([p.detach()], st["sparsity"], self.prune_n, self.prune_m,
                             [init_importance], comparison_group=self.comparison_group)[0]
        st["split"] = z0.detach().clone().to(device=p.device, dtype=self.split_dtype)
        st["initial_split"] = z0.detach().ne(0).clone().to(device=p.device)

        # Gradient snapshot
        st["last_grad_for_importance"] = None

    @torch.no_grad()
    def _proximal_update(self):
        """
        If not decouple:
            Add proximal term to gradients AFTER global gradient clipping and
            BEFORE the actual optimizer step. This ensures proximal is not clipped.
            We also scale proximal to match distributed gradient averaging.
        If decouple:
            Directly update weights with proximal term using the main learning rate.
            This happens AFTER the main optimizer step.
        """
        # Determine world size for average scaling (DDP/FSDP usually average grads across ranks)
        if self.accelerator is not None and getattr(self.accelerator, "num_processes", None):
            world = int(self.accelerator.num_processes)
        elif dist.is_initialized():
            world = dist.get_world_size()
        else:
            world = 1
        avg_div = world if world > 0 else 1

        for g in self.param_groups:
            if not g.get("admm", False):
                continue
            for w in g["params"]:
                if w.grad is None and not self.decouple:
                    continue
                self._lazy_init_admm_state(w, g)
                st = self.state[w]
                dual, split = st["dual"], st["split"]
                lmda = st["lmda"] # Use per-parameter lmda
            
                # Proximal term: λ (w - z + u)
                if self.decouple:# Decoupled: direct weight update, happens AFTER optimizer step
                    ## exact proximal operator
                    # v = st['exp_avg_sq']                                  # v_t
                    # step = int(st.get("step", 0))                         # Adam's current step t
                    # beta2 = g.get('betas', (0.9, 0.999))[1]
                    # eps = g.get('eps', 1e-8)
                    # lr = g.get('lr', 0.0)

                    # # bias correction for second moment: vhat = v_t / (1 - beta2^t)
                    # bc2 = 1.0 - (beta2 ** max(step, 1))                   # guard t=0
                    # vhat = v / max(bc2, 1e-12)

                    # # eta_vec = lr / (sqrt(vhat) + eps) ## effective lr (per element)
                    # # keep dtype/device aligned with w
                    # denom_adam = vhat.sqrt().add_(eps)                    # tmp tensor
                    # eta_vec = (lr / denom_adam).to(dtype=w.dtype)

                    # # exact prox: w <- (w + (lambda*eta_vec)*(z - u)) / (1 + lambda*eta_vec)
                    # a = (split - dual).to(dtype=w.dtype, device=w.device)
                    # gamma = lmda * eta_vec                                # elementwise
                    # w.add_(gamma * a)
                    # w.mul_(1.0 / (1.0 + gamma))
                    ##linearization
                    prox = lmda * (w.detach() - split.detach() + dual.detach())
                    # w.data.add_(prox, alpha=-g['lr']) ## w_k+1/2 = w_k - \eta \lambda (w-z+u)
                    w.data.add_(-prox) ## lr decoupling (constant penalty)
                else:
                    prox = lmda * (w.detach() - split.detach() + dual.detach())
                    # Coupled: add to gradient, happens BEFORE optimizer step
                    # Match AMP grad dtype and distributed averaging scale
                    prox_local = _loc(prox)
                    prox_local = prox_local.to(w.grad.dtype)
                    if avg_div > 1:
                        prox_local = prox_local / avg_div

                    # Add to gradient (handle grad DTensor if any)
                    if hasattr(w.grad, "to_local"):
                        gl = w.grad.to_local()
                        gl.add_(prox_local)
                    else:
                        w.grad.add_(prox_local)

                    # Stash grad for gradient-importance if needed (pre-step, post-clip)
                    if self.projection_mode == "gradient":
                        st["last_grad_for_importance"] = (_loc(w.grad).detach().clone()
                                                          if hasattr(w.grad, "to_local")
                                                          else w.grad.detach().clone())

    @torch.no_grad()
    def _dual_update(self):
        """
        Every 'interval' steps, update split (z) and dual (u), and compute mask_diff.
        - z^{k+1} = Proj(w + u)
        - u^{k+1} = u + α (w - z^{k+1})
        Also compute global mask flip ratio between old z and new z.
        """
        if (self.current_step % self.interval) != 0:
            return

        self.mask_metrics = {'step_hamming': 0.0, 'initial_hamming': 0.0, 'step_iou': 0.0, 'initial_iou': 0.0}
        admm_groups = 0

        for g in self.param_groups:
            if not g.get("admm", False):
                continue
            admm_groups += 1
            weights = list(g["params"])
            if not weights:
                continue

            device = weights[0].device
            flip_sum_step = torch.tensor(0, device=device, dtype=torch.int64)
            flip_sum_initial = torch.tensor(0, device=device, dtype=torch.int64)
            intersection_step = torch.tensor(0, device=device, dtype=torch.int64)
            union_step = torch.tensor(0, device=device, dtype=torch.int64)
            intersection_initial = torch.tensor(0, device=device, dtype=torch.int64)
            union_initial = torch.tensor(0, device=device, dtype=torch.int64)
            numel_sum = torch.tensor(0, device=device, dtype=torch.int64)

            for w in weights:
                self._lazy_init_admm_state(w, g)
                st = self.state[w]

                dual  = st["dual"]
                split = st["split"]
                initial_split = st["initial_split"]
                spars = st["sparsity"]
                current_lmda = st["lmda"]
                previous_lmda = st["prev_lmda"]

                if current_lmda != previous_lmda:
                    dual.mul_(previous_lmda / current_lmda)

                importance_i = None
                if self.projection_mode == "gradient":
                    gsnap = st.get("last_grad_for_importance", None)
                    if gsnap is not None:
                        gi = _loc(gsnap).float()
                        if gi.ndim >= 2:
                            imp_now = (gi * gi).sum(dim=0)
                        else:
                            imp_now = gi.abs()
                        if self.importance_ema > 0.0:
                            st["importance"].mul_(self.importance_ema).add_(imp_now, alpha=1 - self.importance_ema)
                        else:
                            st["importance"].copy_(imp_now)
                        importance_i = st["importance"]
                        st["last_grad_for_importance"] = None
                elif self.projection_mode == "activation":
                    importance_i = st.get("importance", None)
                elif self.projection_mode == "momentum":
                    v_t = st.get("exp_avg_sq")
                    beta2 = g.get('betas', (0.9, 0.95))[1]
                    importance_i = v_t / (1.0 - beta2**(st.get("step", 1)))
                    if isinstance(importance_i, DTensor):
                        # Ensure importance_i is globally consistent for projection
                        # This will gather the sharded exp_avg_sq to all ranks
                        importance_i = importance_i.redistribute(placements=[Replicate()]).to_local()
                elif self.projection_mode == "taylor":
                    m_t = st.get("exp_avg")
                    v_t = st.get("exp_avg_sq")
                    step = st.get("step", 1)
                    beta1, beta2 = g.get('betas', (0.9, 0.999))
                    
                    m_hat = m_t / (1.0 - beta1**step)
                    v_hat = v_t / (1.0 - beta2**step)
                    importance_i = 1.0/2.0 * v_hat * (w.detach()**2) - m_hat*w.detach()
                    # importance_i = (torch.sqrt(v_hat) * w.detach() - m_hat / torch.sqrt(v_hat))

                    if isinstance(importance_i, DTensor):
                        importance_i = importance_i.redistribute(placements=[Replicate()]).to_local()

                is_direct_score = (self.projection_mode == 'taylor')

                z_in  = (w.detach() + dual.detach())
                z_new = self.projection([z_in], spars, self.prune_n, self.prune_m,
                                        [importance_i], comparison_group=self.comparison_group, is_direct_score=is_direct_score)[0]
                z_new = z_new.detach().clone().to(w.device)

                u_new = dual.detach() + self.alpha * (w.detach() - z_new)

                w_l = _loc(w)
                s_l = _loc(split)
                d_l = _loc(dual)
                z_new_l = _loc(z_new)

                new_lmda_for_param = current_lmda
                if self.lmda_schedule_mode == 'adaptive_boyd':
                    r_primal_norm = torch.norm(w_l.detach() - z_new_l.detach())
                    r_dual_norm = current_lmda * torch.norm(z_new_l.detach() - s_l.detach())
                    if r_primal_norm > self.mu * r_dual_norm:
                        new_lmda_for_param = current_lmda * self.tau_incr
                    elif r_dual_norm > self.mu * r_primal_norm:
                        new_lmda_for_param = current_lmda / self.tau_decr
                else:
                    t = self.current_step
                    T = self.total_steps
                    s0 = self.init_lmda
                    s1 = self.final_lmda

                    if self.lmda_schedule_mode == 'constant':
                        new_lmda_for_param = self.lmda_default
                    elif self.lmda_schedule_mode == 'linear':
                        new_lmda_for_param = s0 + (s1 - s0) * (t / T)
                    elif self.lmda_schedule_mode == 'cosine':
                        new_lmda_for_param = s0 + (s1 - s0) * 0.5 * (1 - math.cos(math.pi * t / T))
                    elif self.lmda_schedule_mode == 'log':
                        eps = 1e-6
                        log_t = math.log(t + eps)
                        log_T = math.log(T + eps)
                        new_lmda_for_param = s0 + (s1 - s0) * (log_t / log_T)
                    elif self.lmda_schedule_mode == 'exponential':
                        if s0 <= 0 or s1 <= 0:
                            raise ValueError("For exponential lambda schedule, both init_lmda and final_lmda must be positive.")
                        new_lmda_for_param = s0 * (s1/s0)**(t/T)

                old_local = _loc(split)
                new_local = _loc(z_new)
                initial_local = _loc(initial_split)

                old_mask = (old_local != 0)
                new_mask = (new_local != 0)
                initial_mask = initial_local

                flip_local_step = (old_mask ^ new_mask).sum().to(device=device)
                flip_local_initial = (initial_mask ^ new_mask).sum().to(device=device)
                numel_local = torch.tensor(old_local.numel(), device=device)

                intersection_step += (old_mask & new_mask).sum().to(device=device)
                union_step += (old_mask | new_mask).sum().to(device=device)
                intersection_initial += (initial_mask & new_mask).sum().to(device=device)
                union_initial += (initial_mask | new_mask).sum().to(device=device)

                flip_sum_step += flip_local_step
                flip_sum_initial += flip_local_initial
                numel_sum += numel_local

                dual.copy_(u_new)
                split.copy_(z_new)
                st["lmda"] = new_lmda_for_param
                st["prev_lmda"] = current_lmda

            if dist.is_initialized():
                dist.all_reduce(flip_sum_step,  op=dist.ReduceOp.SUM)
                dist.all_reduce(flip_sum_initial, op=dist.ReduceOp.SUM)
                dist.all_reduce(intersection_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(union_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(intersection_initial, op=dist.ReduceOp.SUM)
                dist.all_reduce(union_initial, op=dist.ReduceOp.SUM)
                dist.all_reduce(numel_sum, op=dist.ReduceOp.SUM)

            eps = 1e-12
            self.mask_metrics['step_hamming'] += float(flip_sum_step.float() / (numel_sum.float() + eps))
            self.mask_metrics['initial_hamming'] += float(flip_sum_initial.float() / (numel_sum.float() + eps))
            self.mask_metrics['step_iou'] += float(intersection_step.float() / (union_step.float() + eps))
            self.mask_metrics['initial_iou'] += float(intersection_initial.float() / (union_initial.float() + eps))

        if admm_groups > 0:
            self.mask_metrics['step_hamming'] /= admm_groups
            self.mask_metrics['initial_hamming'] /= admm_groups
            self.mask_metrics['step_iou'] /= admm_groups
            self.mask_metrics['initial_iou'] /= admm_groups

    @torch.no_grad()
    def step(self, closure=None):
        """
        Order depends on decouple:
        - Coupled (default):
            1) (Trainer did backward and clipping)
            2) _proximal_update() adds proximal term to grad
            3) super().step() uses combined grad
            4) _dual_update() for z/u
        - Decoupled:
            1) (Trainer did backward and clipping)
            2) super().step() uses original grad
            3) _proximal_update() applies proximal term directly to weights
            4) _dual_update() for z/u
        """
        if not self.decouple:
            self._proximal_update()

        out = super().step(closure)

        if self.decouple:
            self._proximal_update()

        self._dual_update()
        self.current_step += 1
        return out

    @torch.no_grad()
    def final_projection(self):
        """
        Apply the final projection to ADMM-tagged parameter groups (in-place).
        This should be called after training is complete to ensure weights have the desired sparsity structure.
        """
        for g in self.param_groups:
            if not g.get("admm", False):
                continue
            for w in g["params"]:
                if w not in self.state or len(self.state[w]) == 0:
                    self._lazy_init_admm_state(w, g)

                st = self.state[w]
                importance = None
                is_direct_score = (self.projection_mode == 'taylor')

                if self.projection_mode == "gradient" or self.projection_mode == "activation":
                    importance = st.get("importance", None)
                elif self.projection_mode == "momentum":
                    v_t = st.get("exp_avg_sq")
                    beta2 = g.get('betas', (0.9, 0.95))[1]
                    importance = v_t / (1.0 - beta2**(st.get("step", 1)))
                    if isinstance(importance, DTensor):
                        importance = importance.redistribute(placements=[Replicate()]).to_local()
                elif self.projection_mode == "taylor":
                    m_t = st.get("exp_avg")
                    v_t = st.get("exp_avg_sq")
                    step = st.get("step", 1)
                    beta1, beta2 = g.get('betas', (0.9, 0.999))
                    m_hat = m_t / (1.0 - beta1**step)
                    v_hat = v_t / (1.0 - beta2**step)
                    importance_i = 1.0/2.0 * v_hat * (w.detach()**2) - m_hat*w.detach()
                    # importance = (torch.sqrt(v_hat) * w.detach() - m_hat / torch.sqrt(v_hat))
                    if isinstance(importance, DTensor):
                        importance = importance.redistribute(placements=[Replicate()]).to_local()

                wnew = self.projection([w.detach()], st["sparsity"], self.prune_n, self.prune_m,
                                       [importance], comparison_group=self.comparison_group, is_direct_score=is_direct_score)[0]
                w.data.copy_(wnew)

    def get_mask_metrics(self) -> Dict[str, float]:
        """
        Return the averaged mask metrics computed at the last interval update.
        """
        return self.mask_metrics

    def get_lmda_stats(self) -> Dict[str, float]:
        """
        Calculates and returns statistics (average, min, max) of per-parameter lmda values.
        """
        total_lmda = 0.0
        count = 0
        min_lmda = float('inf')
        max_lmda = float('-inf')

        for g in self.param_groups:
            if not g.get("admm", False):
                continue
            for w in g["params"]:
                if w in self.state:
                    lmda_val = self.state[w].get("lmda")
                    if lmda_val is not None:
                        total_lmda += lmda_val
                        count += 1
                        min_lmda = min(min_lmda, lmda_val)
                        max_lmda = max(max_lmda, lmda_val)
        
        if count == 0:
            return {"avg_lmda": 0.0, "min_lmda": 0.0, "max_lmda": 0.0}
        else:
            return {"avg_lmda": total_lmda / count, "min_lmda": min_lmda, "max_lmda": max_lmda}


class MaskedAdam(torch.optim.Adam):
    """
    A variant of Adam that applies a fixed mask to the parameters after each
    optimizer step. This is useful for retraining pruned models, ensuring that
    the pruned weights remain zero.
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.masks = []
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.dim() > 1: # Typically, we only prune weights, not biases
                        self.masks.append((p.data != 0).clone())
                    else:
                        self.masks.append(None)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step and then re-applies the sparsity mask.
        """
        super().step(closure)
        
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if self.masks[i] is not None:
                    p.data.mul_(self.masks[i])
                i += 1


## LEGACY ADMM
# class ADMM(torch.optim.Optimizer):
#     def __init__(
#         self, 
#         param_groups,
#         projection_fn,
#         sparsity: float,
#         interval: int,
#         base_optimizer: torch.optim.Optimizer = torch.optim.SGD,
#         alpha: float = 1.0,
#         lmda: float = 1e-3,
#         lr: float = 2e-4,
#         prune_n: int = 0,
#         prune_m: int = 0,
#         # importance_matrix is removed from __init__
#         comparison_group: str = 'layer',
#         projection_mode: str = 'identity',
#         importance_ema: float = 0.0,
#         **kwargs
#     ):
#         """
#         ADMM optimizer with both sparsity and importance matrix stored in optimizer state.
#         Args:
#             param_groups (list): List of parameter groups.
#                 Each group is a dict, e.g., {'params': [...], 'admm': True}
#                 'admm': True indicates this group's params are subject to ADMM.
#             projection_fn (callable): Projection function to use.
#             sparsity (float): Sparsity target.
#             interval (int): Interval for dual update.
#             base_optimizer (torch.optim.Optimizer): Base optimizer to use.
#             alpha (float): Over-relaxation parameter for ADMM.
#             lmda (float): Penalty parameter.
#             lr (float): Learning rate for the base optimizer.
#             prune_n (int): n for n:m structured sparsity.
#             prune_m (int): m for n:m structured sparsity.
#             importance_matrix (list[torch.Tensor], optional): Importance matrix for projection.
#             comparison_group (str): Comparison group for projection ('layer', 'column', 'row').
#             projection_mode (str): Mode for the projection function ('identity', 'gradient', 'activation').
#             importance_ema (float): EMA coefficient for importance matrix.
#             **kwargs: Additional arguments for the base optimizer.
#         """
#         if not callable(projection_fn):
#             raise TypeError("projection_fn must be a callable function.")
        
#         defaults = dict(lr=lr, **kwargs)
#         super(ADMM, self).__init__(param_groups, defaults)

#         self.projection = projection_fn
#         self.comparison_group = comparison_group.lower()
#         if self.comparison_group not in ['layer', 'column', 'row']:
#             raise ValueError(f"comparison_group must be one of 'layer', 'column', 'row'. Got {self.comparison_group}.")
        
#         self.alpha = alpha
#         if not (0 <= self.alpha <= 2):
#             raise ValueError(f"alpha must be in the range [0, 2]. Got {self.alpha}.")
        
#         self.projection_mode = projection_mode.lower()
#         if self.projection_mode not in ['identity', 'gradient', 'activation']:
#             raise ValueError(f"projection_mode must be one of 'identity', 'gradient', 'activation'. Got {self.projection_mode}.")
        
#         self.importance_ema = importance_ema
#         self.sparsity = sparsity # Global default sparsity
#         self.interval = interval
#         self.current_step = 0
#         self.prune_n = prune_n
#         self.prune_m = prune_m
#         self.mask_diff = 0.0

#         for group in self.param_groups:
#             if group.get('admm', False):
#                 group['lmda'] = lmda
        
#         base_param_groups = []
#         for pg in self.param_groups:
#             base_pg = {k: v for k, v in pg.items() if k not in ['lmda', 'admm']}
#             base_param_groups.append(base_pg)
#         self.base_optimizer = base_optimizer(base_param_groups, **kwargs)

#     def _lazy_init_admm_state(self, p: torch.nn.Parameter):
#         """
#         Lazily initialize the 'dual', 'split', 'sparsity', and 'importance' states for a parameter.
#         """
#         if 'dual' in self.state[p]:
#             return

#         st = self.state[p]
#         st['dual'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#         st['sparsity'] = self.sparsity
#         init_importance = None
#         if self.projection_mode != 'identity': 
#             #TODO: implement here
#             # Store only the diagonal of the importance matrix as a vector.
#             # For a Linear layer weight of shape (out, in), the diagonal has 'in' elements.
#             st['importance'] = torch.zeros(p.shape[-1], device=p.device)
#             init_importance = st['importance']

#         # Initial projection for 'split' uses the initial importance (zeros)
#         z0 = self.projection(
#             [p.detach()], st['sparsity'], self.prune_n, self.prune_m, [init_importance], comparison_group=self.comparison_group
#         )[0]
#         st['split'] = z0.detach().clone().to(device=p.device)  
    
#     def final_projection(self):
#         """Applies the final projection to the ADMM parameters."""
#         with torch.no_grad():
#             for group in self.param_groups:
#                 if group.get('admm', False):
#                     for w in group['params']:
#                         self._lazy_init_admm_state(w)
                        
#                         p_sparsity = self.state[w]['sparsity']
#                         p_importance = None
#                         if self.projection_mode != 'identity':
#                             p_importance = self.state[w]['importance']
                        
#                         final_weight = self.projection(
#                             [w.detach()], p_sparsity, self.prune_n, self.prune_m, [p_importance], self.comparison_group
#                         )[0]
#                         w.data.copy_(final_weight)

#     @torch.no_grad()
#     def step(self, zero_grad=False):
#         for group in self.param_groups:
#             if group.get('admm', False):
#                 weights = group['params']
#                 lmda = group['lmda']
#                 for w in weights:
#                     if w.grad is None: continue
#                     self._lazy_init_admm_state(w)
#                     dual = self.state[w]['dual']
#                     split = self.state[w]['split']
#                     proximal = lmda * (w.detach() - split.detach() + dual.detach())
#                     w.grad.add_(proximal)

#         self.base_optimizer.step()
  
#         if (self.current_step + 1) % self.interval == 0:
#             self.mask_diff = 0.0
#             with torch.no_grad():
#                 for group in self.param_groups:
#                     if group.get('admm', False):
#                         weights = group['params']
#                         lmda = group['lmda']
#                         for w in weights:
#                             st = self.state[w]
#                             dual = st['dual']
#                             split = st['split']
#                             p_sparsity = st['sparsity']

#                             z_input_i = w.detach() + dual.detach()
#                             importance_matrix_i = None
                            
#                             if self.projection_mode == 'gradient':
#                                 proximal = lmda * (w.detach() - split.detach() + dual.detach())
#                                 grad_input_i = w.grad.detach() - proximal
#                                 current_importance = torch.pow(grad_input_i, 2)
                                
#                                 # Update importance matrix in the state using EMA
#                                 if self.importance_ema > 0:
#                                     st['importance'].mul_(self.importance_ema).add_(current_importance, alpha=1 - self.importance_ema)
#                                 else:
#                                     st['importance'].copy4_(current_importance)
#                                 importance_matrix_i = st['importance']
                            
#                             elif self.projection_mode == 'activation':
#                                 # For activation mode, the trainer should have updated the state beforehand.
#                                 # We just use the value from the state.
#                                 importance_matrix_i = st['importance']

#                             z_new_i = self.projection(
#                                 [z_input_i], p_sparsity, self.prune_n, self.prune_m,
#                                 [importance_matrix_i], comparison_group=self.comparison_group
#                             )[0]
                            
#                             u_new_i = dual.detach() + self.alpha * (w.detach() - z_new_i)
                            
#                             ## if Distributed
#                             if isinstance(split,DTensor):
#                                 old_mask_local = split.to_local()
#                                 new_mask_local = z_new_i.to_local()

#                                 old_zero = (old_mask_local == 0)
#                                 new_zero = (new_mask_local == 0)
                                
#                                 flip_local = (old_zero ^ new_zero).sum()
#                                 numel_local = torch.tensor(old_mask_local.numel(),device = old_mask_local.device)

#                                 dist.all_reduce(flip_local, op=dist.ReduceOp.SUM)
#                                 dist.all_reduce(numel_local, op=dist.ReduceOp.SUM)

#                             else:
#                                 old_mask = (split == 0).detach().cpu()
#                                 new_mask = (z_new_i == 0).detach().cpu()

#                                 flip_local = (old_mask ^ new_mask).sum()
#                                 numel_local = torch.tensor(old_mask.numel(), device=old_mask.device)

#                             mask_diff = (flip_local.float() / numel_local.float()).item()
#                             self.mask_diff += mask_diff

#                             dual.copy_(u_new_i)
#                             split.copy_(z_new_i)
                        
#                         if len(weights) > 0:
#                             self.mask_diff /= len(weights)
                            
#         self.current_step += 1

#     def get_mask_diff(self):
#         """
#         Returns the mask difference since the last step.
#         This is useful for logging or monitoring changes in sparsity.
#         """
#         return self.mask_diff

### TODO: UPDATE SAFE
class SAFE(torch.optim.Optimizer):
    def __init__(self, 
                param_groups,
                projection_fn,
                sparsity: float,
                interval: int,
                base_optimizer: torch.optim.Optimizer = torch.optim.SGD,
                alpha: float = 1.0,
                lmda: float = 1e-3,
                lr: float = 2e-4,
                rho: float = 0.05,
                prune_n: int = 0,
                prune_m: int = 0,
                importance_matrix: list[torch.Tensor] = None,
                comparison_group: str = 'layer',
                param_names: Dict[torch.nn.Parameter, str] = None,
                **kwargs):
        """
        SAFE optimizer 
        Args:
            param_groups (list): List of parameter groups.
                Each group is a dict, e.g., {'params': [...], 'admm': True, 'lmda': 0.01}
                'admm': True indicates this group's params are subject to ADMM. Dual and split variables are created as optimizer state for these params.
                'lmda': Penalty parameter for this ADMM group.
            projection_fn (callable): Projection function to use. Should take a list of tensors and return a list of projected tensors.
                Expected signature: projection_fn(params_list, sparsity, prune_n, prune_m, importance_matrix) -> projected_params_list
            sparsity (float): Sparsity target.
            interval (int): Interval for dual update.
            base_optimizer (torch.optim.Optimizer): Base optimizer to use for SAM.
            alpha (float): Over-relaxation parameter for ADMM. Default is 1.0.
            lmda (float): penalty parameter.
            lr (float): Learning rate for the base optimizer.
            rho (float): Perturbation size for SAM.
            prune_n (int): n for n:m structured sparsity.
            prune_m (int): m for n:m structured sparsity.
            importance_matrix (list[torch.Tensor], optional): Importance matrix used for generalized projection. Must have the same structure as param_groups with admm=True.
            comparison_group (str): Comparison group for ADMM projection ('layer', 'column', 'row').
            **kwargs: Additional arguments for the base optimizer.
        """
        if not callable(projection_fn):
            raise TypeError("projection_fn must be a callable function.")
        self.projection= projection_fn
        self.comparison_group = comparison_group.lower()
        if self.comparison_group not in ['layer', 'column', 'row']:
            raise ValueError(f"comparison_group must be one of 'layer', 'column', 'row'. Got {self.comparison_group}.")
        processed_param_groups = []
        for i, group in enumerate(param_groups):
            if group.get('admm', False):
                if not group['params']: # Should not happen if group is valid
                    print(f"Warning: ADMM group {i} has no params.")
                    processed_param_groups.append(group)
                    continue
                admm_params_list = group['params'] # This should be a list of tensors

                group['duals'] = [torch.zeros_like(p, device=p.device) for p in admm_params_list]
                if importance_matrix is not None:
                    if len(importance_matrix) != len(admm_params_list):
                        raise ValueError(f"importance_matrix must have the same length as params in group {i}.")
                group['splits'] = self.projection(admm_params_list, sparsity, prune_n, prune_m, importance_matrix, comparison_group=self.comparison_group)

                if 'lmda' not in group:
                    group['lmda'] = lmda
            processed_param_groups.append(group)
        
        defaults = dict(lr=lr, rho=rho, **kwargs) # lmda is now per-group
        super(SAFE, self).__init__(processed_param_groups, defaults)
        sam_param_groups = []
        for pg in self.param_groups: # self.param_groups is now processed_param_groups
            sam_pg = {k: v for k, v in pg.items() if k not in ['duals', 'splits', 'admm', 'lmda']}
            sam_param_groups.append(sam_pg)
        self.base_optimizer = SAM(sam_param_groups, base_optimizer, rho=rho, **kwargs)

        ## other control variables
        self.alpha = alpha
        if not (0 <= self.alpha <= 2):
            raise ValueError(f"alpha must be in the range [0, 2]. Got {self.alpha}.")
        self.importance_matrix = importance_matrix if importance_matrix is not None else None
        self.sparsity = sparsity
        self.interval = interval
        self.current_step = 0
        self.prune_n = prune_n
        self.prune_m = prune_m
        
    def update_importance_matrix(self, importance_matrix):
        if len(importance_matrix) != len(self.param_groups):
            raise ValueError("importance_matrix must have the same length as param_groups.")
        self.importance_matrix = importance_matrix
    
    def final_projection(self):
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
            final_weights = self.projection(
                weights,
                self.sparsity,
                prune_n=self.prune_n,
                prune_m=self.prune_m,
                importance_matrix=self.importance_matrix,
                comparison_group=self.comparison_group
            )
            for w,fw in zip(weights,final_weights):
                w.data.copy_(fw)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.base_optimizer.first_step(zero_grad)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # get x_t,u_t,z_t
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
                lmda = group['lmda']
                duals = group['duals']
                splits = group['splits']

                for i in range(len(weights)):
                    proximal = lmda * (weights[i].detach() - splits[i].detach() + duals[i].detach()) # proximal gradient w-z+u
                    weights[i].grad.add_(proximal) 
        self.base_optimizer.second_step(zero_grad)
        
        # dual update
        if (self.current_step + 1) % self.interval == 0:
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('admm', False):
                        weights = group['params'] # W_t+1
                        duals = group['duals']   # U_t
                        splits = group['splits'] # Z_t

                        if not all([weights, duals, splits]):
                            print(f"Warning: ADMM group missing weights, duals, or splits. Skipping dual/split update for this group.")
                            continue
                        if not (len(weights) == len(duals) == len(splits)):
                            print(f"Warning: Mismatch in lengths of weights, duals, splits for an ADMM group. Skipping dual/split update.")
                            continue
                        
                        for i in range(len(duals)):
                            z_input_i = weights[i].detach() + duals[i].detach()

                            z_new_i = self.projection(
                                [z_input_i],
                                sparsity,
                                prune_n=self.prune_n,
                                prune_m=self.prune_m,
                                importance_matrix=self.importance_matrix,
                                comparison_group=self.comparison_group
                            )[0]

                            u_new_i = duals[i].detach() + self.alpha * (weights[i].detach() - z_new_i) # U_t+1 = U_t + \alpha(W_t+1 - Z_t+1)
                            
                            duals[i].copy_(u_new_i)
                            splits[i].copy_(z_new_i)
                        # z_input = [w.detach() + u.detach() for w, u in zip(weights, duals)] # proj (W_t+1 + U_t)
                        # z_new = self.projection(z_input, sparsity, prune_n=self.prune_n, prune_m=self.prune_m, importance_matrix=self.importance_matrix)
                        # u_new = [u.detach() + w.detach() - z_n.detach() for u, w, z_n in zip(duals, weights, z_new)] # U_t+1 = U_t + W_t+1 - Z_t+1
                        
                        # for i in range(len(duals)): # Iterate using index to ensure correct assignment
                        #     duals[i].copy_(u_new[i])
                        #     splits[i].copy_(z_new[i])
        
        self.current_step += 1
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAFE requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

class SAM(torch.optim.Optimizer):
    def __init__(
        self, 
        params,
        base_optimizer: torch.optim.Optimizer,
        rho:int =0.05,
        adaptive:bool =False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        """
        SAM optimizer. Implementation from https://github.com/davda54/sam (SAM)
        Args:
            params (iterable): Parameters to optimize or dicts defining parameter groups.
            base_optimizer (torch.optim.Optimizer): Base optimizer to use.
            rho (float): Perturbation size for SAM.
            adaptive (bool): Whether to use adaptive scaling.
            **kwargs: Additional arguments for the base optimizer.
        """

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            grad_norm = self._grad_norm()
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
