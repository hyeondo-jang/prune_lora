import math
from absl import logging

##TODO: Fix schedulers for state-ful admm compatibility, and distributed training
class PenaltyScheduler:

    def __init__(self, optimizer,initial_lmda, final_lmda, total_steps, mu=1.5, tau_incr=2, tau_decr=2, mode='linear', splits=None):
        """
        A scheduler for penalty parameter (lmda).
        
        Args:
            optimizer (torch.optim.Optimizer) : optimizer for which the scheduler is used
            initial_lmda (float): initial value of lambda
            final_lmdas (float): final value of lambda
            total_steps (int): Total number of steps for scheduling.
            mode (str): Scheduling mode ( 'constant','linear','cosine','adaptive').
        """
        self.optimizer = optimizer
        self.total_steps = total_steps

        self.mode = mode.lower()
        self.initial_lmda = initial_lmda
        self.final_lmda = final_lmda
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr
        self.step_count = 0

    def step(self, r_primal_norms=None, r_dual_norms=None, supp_change_rates=None):
        """Update the scheduler step."""
        self.step_count += 1
        self.step_count = min(self.step_count, self.total_steps)  # Clamp to total_steps

        if self.mode == 'adaptive_boyd':
            self._update_adaptive_lmda(r_primal_norms, r_dual_norms)
        else:
            new_lmda = self._calculate_lmda_for_fixed_modes()
            for group in self.optimizer.param_groups:
                if group.get('admm', False):
                    group['lmda'] = new_lmda
                    # logging.info(f"Updated lmda to {new_lmda} for group {group['name']} at step {self.step_count}")
            
    def _calculate_lmda_for_fixed_modes(self):
        """Calculate the current lmda based on the mode for non-adaptive modes."""
        init_lmda = self.initial_lmda
        final_lmda = self.final_lmda

        if self.mode == 'linear':
            return init_lmda + (final_lmda - init_lmda) * (self.step_count / self.total_steps)
        
        elif self.mode == 'constant':
            return final_lmda
        
        elif self.mode == 'cosine':
            return init_lmda + (final_lmda - init_lmda) * 0.5 * (1 - math.cos(math.pi * self.step_count / self.total_steps))
        
        elif self.mode == 'log':
            log_step = math.log(1 + self.step_count)
            log_total = math.log(1 + self.total_steps)
            return init_lmda + (final_lmda - init_lmda) * (log_step / log_total)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _update_adaptive_lmda(self, r_primal_norms, r_dual_norms):
        """Update lmda for each ADMM group based on adaptive Boyd's scheme."""
        assert r_primal_norms is not None and r_dual_norms is not None, "adaptive_boyd requires residuals"

        for group in self.optimizer.param_groups:
            if group.get('admm', False):
                current_lmda = group.get('lmda', self.initial_lmda) # Use initial_lmda as fallback
                
                new_lmda = current_lmda
                # Ensure residuals are tensors and on the same device for comparison
                # Assuming r_primal_norms and r_dual_norms are already global sums
                if r_primal_norms.item() > self.mu * r_dual_norms.item():
                    new_lmda = current_lmda * self.tau_incr
                elif r_dual_norms.item() > self.mu * r_primal_norms.item():
                    new_lmda = current_lmda / self.tau_decr
                
                # Clamp the new value
                new_lmda = max(min(new_lmda, self.final_lmda), 1e-4)
                group['lmda'] = new_lmda
                # logging.info(f"Updated lmda to {new_lmda} for group {group['name']} at step {self.step_count}")
        
class SparsityScheduler:
    def __init__(self, optimizer, initial_sparsity, final_sparsity, start_step, final_step, mode='cubic'):
        """
        Scheduler for gradually increasing sparsity over training steps.

        Args:
            optimizer (torch.optim.Optimizer): optimizer for which the scheduler is used
            initial_sparsity (float): initial sparsity (0 ~ 1).
            final_sparsity (float): final sparsity (0 ~ 1).
            start_step (int): start step.
            final_step (int): final step.
            mode (str): scheduling mode - 'constant', 'linear', 'cosine', 'exponential'.
        """
        self.optimizer = optimizer
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.final_step = final_step
        self.mode = mode.lower()
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.step_count = min(self.step_count, self.final_step)
        self.optimizer.sparsity = self.get_sparsity()
        # logging.info(f"Updated sparsity to {self.optimizer.sparsity} at step {self.step_count}")


    def get_sparsity(self):
        t0 = self.start_step
        t = self.step_count
        T = self.final_step
        s0 = self.initial_sparsity
        s1 = self.final_sparsity

        if self.mode == 'constant':
            return s1
        elif self.mode == 'linear':
            if t < t0:
                return 0
            else:
                return s0 + (s1 - s0) * ((t-t0) / (T-t0))
        elif self.mode == 'cosine':
            if t < t0:
                return 0
            else:
                return s0 + (s1 - s0) * 0.5 * (1 - math.cos(math.pi * (t-t0) / (T-t0)))
        elif self.mode == 'exponential':
            if t < t0:
                return 0
            else:
                power = (t-t0) / (T-t0)
                return s0 * (s1 / s0) ** power if s0 > 0 else s1 * power
        elif self.mode == 'log':
            if t < t0:
                return 0
            else:
                # Avoid log(0); add epsilon
                eps = 1e-6
                log_t = math.log((t-t0) + eps)
                log_T = math.log((T-t0) + eps)
                return s0 + (s1 - s0) * (log_t / log_T)
        elif self.mode == 'cubic':
            if t < t0:
                return 0
            else:
                return s1 + (s0 - s1) * (1 - (t-t0) / (T-t0)) ** 3
        else:
            raise ValueError(f"Unknown scheduling mode: {self.mode}")