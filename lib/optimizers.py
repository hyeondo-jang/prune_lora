import torch
from typing import Union, Dict

class ADMM(torch.optim.Optimizer):
    def __init__(
        self, 
        param_groups,
        projection_fn,
        sparsity: Union[float, Dict[int, float]], # sparsity를 float 또는 dict로 받을 수 있도록 수정
        interval: int,
        base_optimizer: torch.optim.Optimizer = torch.optim.SGD,
        alpha: float = 1.0,
        lmda: float = 1e-3,
        lr: float = 2e-4,
        prune_n: int = 0,
        prune_m: int = 0,
        importance_matrix: list[torch.Tensor] = None,
        comparison_group: str = 'layer',
        projection_mode: str = 'identity',
        importance_ema: float = 0.0,
        **kwargs
    ):
        """
        ADMM optimizer 
        Args:
            param_groups (list): List of parameter groups.
                Each group is a dict, e.g., {'params': [...], 'admm': True, 'lmda': 0.01}
                'admm': True indicates this group's params are subject to ADMM. Dual and split variables are created as optimizer state for these params.
                'lmda': Penalty parameter for this ADMM group.
            projection_fn (callable): Projection function to use. Should take a list of tensors and return a list of projected tensors.
                Expected signature: projection_fn(params_list, sparsity, prune_n, prune_m, importance_matrix) -> projected_params_list
            sparsity (float): Sparsity target.
            interval (int): Interval for dual update.
            base_optimizer (torch.optim.Optimizer): Base optimizer to use. e.g. torch.optim.Adam, torch.optim.SGD
            alpha (float): Over-relaxation parameter for ADMM. Default is 1.0.
            lmda (float): penalty parameter.
            lr (float): Learning rate for the base optimizer.
            prune_n (int): n for n:m structured sparsity.
            prune_m (int): m for n:m structured sparsity.
            importance_matrix (list[torch.Tensor], optional): Importance matrix used for generalized projection. Must have the same structure as param_groups with admm=True.
            comparison_group (str): Comparison group for ADMM projection ('layer', 'column', 'row').
            projection_mode (str): Mode for the projection function. Default is 'identity (P=I)'. We currently support 'gradient' (P = diag(nabla L nabla L^T)) and 'activation' (P = diag(A^TA)). Note that 'activation' needs importance_matrix to be provided.
            importance_ema (float): Exponential moving average coefficient for importance matrix. Default is 0.0 (no EMA).
            **kwargs: Additional arguments for the base optimizer.
        """
        if not callable(projection_fn):
            raise TypeError("projection_fn must be a callable function.")
        
        # Define defaults and call the parent constructor FIRST.
        # This initializes self.param_groups and self.state.
        defaults = dict(lr=lr, **kwargs)
        super(ADMM, self).__init__(param_groups, defaults)

        self.projection= projection_fn
        self.comparison_group = comparison_group.lower()
        if self.comparison_group not in ['layer', 'column', 'row']:
            raise ValueError(f"comparison_group must be one of 'layer', 'column', 'row'. Got {self.comparison_group}.")

        # Now, iterate through self.param_groups to set up ADMM-specific state and variables.
        for i, group in enumerate(self.param_groups):
            if group.get('admm', False):
                if not group['params']:
                    print(f"Warning: ADMM group {i} has no params.")
                    continue
                
                admm_params_list = group['params']

                # Initialize per-parameter sparsity in the optimizer's state
                for p in admm_params_list:
                    # self.state[p] is now a valid defaultdict
                    self.state[p]['sparsity'] = sparsity if isinstance(sparsity, float) else sparsity.get(id(p), 0.5)

                group['duals'] = [torch.zeros_like(p, device=p.device) for p in admm_params_list]
                if importance_matrix is not None:
                    if len(importance_matrix) != len(admm_params_list):
                        raise ValueError(f"importance_matrix must have the same length as params in group {i}.")
                
                # Use per-parameter sparsity for the initial projection
                splits_list = []
                for p in admm_params_list:
                    # Now this line is safe to call
                    p_sparsity = self.state[p]['sparsity']
                    splits_list.append(self.projection([p], p_sparsity, prune_n, prune_m, importance_matrix, comparison_group=self.comparison_group)[0])
                group['splits'] = splits_list

                if 'lmda' not in group:
                    group['lmda'] = lmda
        
        # Create the base optimizer using the processed param_groups
        base_param_groups = []
        for pg in self.param_groups:
            # Filter out ADMM-specific keys for the base optimizer
            base_pg = {k: v for k, v in pg.items() if k not in ['duals', 'splits', 'admm', 'lmda']}
            base_param_groups.append(base_pg)
        self.base_optimizer = base_optimizer(base_param_groups, **kwargs)

        ## other control variables
        self.alpha = alpha
        if not (0 <= self.alpha <= 2):
            raise ValueError(f"alpha must be in the range [0, 2]. Got {self.alpha}.")
        self.importance_matrix = importance_matrix if importance_matrix is not None else None
        self.projection_mode = projection_mode.lower()
        if self.projection_mode not in ['identity', 'gradient', 'activation']:
            raise ValueError(f"projection_mode must be one of 'identity', 'gradient', 'activation'. Got {self.projection_mode}.")
        self.importance_ema = importance_ema
        # self.sparsity는 이제 사용되지 않음. self.state[p]['sparsity']를 사용.
        self.interval = interval
        self.current_step = 0
        self.prune_n = prune_n
        self.prune_m = prune_m

    def update_importance_matrix(self, importance_matrix):
        if len(importance_matrix) != len(self.param_groups[0]['params']):
            raise ValueError("importance_matrix must have the same length as params in the first ADMM(weight) group.")
        if self.importance_ema > 0:
            if not self.importance_matrix:
                self.importance_matrix = [v.clone().detach() for v in importance_matrix]
            else:
                for i in range(len(self.importance_matrix)):
                    self.importance_matrix[i] = self.importance_ema * self.importance_matrix[i] + (1 - self.importance_ema) * importance_matrix[i].clone().detach()
        else:
            self.importance_matrix = importance_matrix

    def update_sparsity(self, sparsity_map: Dict[int, float]):
        """Updates the sparsity for each parameter from a dictionary."""
        for group in self.param_groups:
            if group.get('admm', False):
                for p in group['params']:
                    if id(p) in sparsity_map:
                        self.state[p]['sparsity'] = sparsity_map[id(p)]

    def final_projection(self):
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
                for i in range(len(weights)):
                    w = weights[i].detach()
                    p_sparsity = self.state[w].get('sparsity', 0.5) # 기본값 제공
                    final_weight=self.projection(
                        [w],
                        p_sparsity,
                        prune_n=self.prune_n,
                        prune_m=self.prune_m,
                        importance_matrix=self.importance_matrix[i] if self.importance_matrix is not None else None,
                        comparison_group=self.comparison_group
                    )[0]
                    weights[i].data.copy_(final_weight)

    @torch.no_grad()
    def step(self, zero_grad=False):
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
        self.base_optimizer.step()
        
        # dual update
        if (self.current_step + 1) % self.interval == 0:
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('admm', False):
                        weights = group['params'] # W_t+1
                        lmda = group['lmda']
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
                            importance_matrix_i = None # 기본값
                            if self.projection_mode == 'gradient':
                                proximal = lmda * (weights[i].detach() - splits[i].detach() + duals[i].detach())
                                grad_input_i = weights[i].grad.detach() - proximal
                                current_importance = torch.pow(grad_input_i, 2)
                                if self.importance_ema <= 0:
                                    importance_matrix_i = current_importance
                                # EMA
                                else:
                                    if not self.importance_matrix:
                                        self.importance_matrix = [torch.zeros_like(p) for p in weights]
                                    beta = self.importance_ema
                                    self.importance_matrix[i].mul_(beta).add_(current_importance, alpha=1 - beta)
                                    importance_matrix_i = self.importance_matrix[i]
                            elif self.projection_mode == 'activation':
                                if self.importance_matrix is not None and i<len(self.importance_matrix):
                                    importance_matrix_i = self.importance_matrix[i].unsqueeze(0).to(z_input_i.device)

                            # 파라미터별 sparsity를 state에서 가져와 사용
                            p_sparsity = self.state[weights[i]].get('sparsity', 0.5)

                            z_new_i = self.projection(
                                [z_input_i],
                                p_sparsity,
                                prune_n=self.prune_n,
                                prune_m=self.prune_m,
                                importance_matrix=importance_matrix_i,
                                comparison_group=self.comparison_group
                            )[0]

                            u_new_i = duals[i].detach() + self.alpha * (weights[i].detach() - z_new_i) # U_t+1 = U_t + \alpha(W_t+1 - Z_t+1)

                            duals[i].copy_(u_new_i)
                            splits[i].copy_(z_new_i)
        self.current_step += 1


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
