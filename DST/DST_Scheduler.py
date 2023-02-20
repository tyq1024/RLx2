from __future__ import annotations
from typing import Dict

import numpy as np
import torch
from torch._C import device

from DST.utils import get_W

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_set(weights:list[torch.Tensor], sparsity:list[float], keep_first_layer_dense:bool=False)->list[float]:
    # Compute the sparsity of each layer

    # We adopt the Erdos Renyi strategy to sparsify the networks.
    # This implementation is based on the open-source repository of the paper 
    # "Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training".
    # Please refer to https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization/blob/main/sparselearning/core.py .

    # This strategy will try to find the right epsilon which makes the following equations hold:
    #            (1-S)*\sum_{l in L}{I_l*O_l} = \sum_{l in L}{(1-S_l)*I_l*O_l}
    #               1-S_l = epsilon*(I_l+O_l)/(I_l*O_l) , l in L
    # where L denotes the indexes of all the layers.
    # However it is possible that one of the sparsity is less than 0, in which case we hold the 
    # sparsity of this layer remains as 0, and reallocation the sparsities in the rest layers.
    # Denote the indexes of the omitted layers as L', the equations become:
    #            (1-S)*\sum_{l in L}{I_l*O_l} = \sum_{l in L}{(1-S_l)*I_l*O_l}
    #               1-S_l = epsilon*(I_l+O_l)/(I_l*O_l) , l in L/L'
    #                                S_l = 0, l in L'

    ans = []
    is_valid = False
    dense_layers = set()
    if keep_first_layer_dense:
        dense_layers.add(0)
    while not is_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for i, w in enumerate(weights):
            n_param = w.numel()
            n_zeros = n_param * sparsity
            n_ones = n_param * (1 - sparsity)

            if i in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[i] = np.sum(w.shape) / w.numel()
                                            
                divisor += raw_probabilities[i] * n_param
        if len(dense_layers) == len(weights): raise Exception('Cannot set a proper sparsity')
        epsilon = rhs / divisor
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_valid = False
            for weight_i, weight_raw_prob in raw_probabilities.items():
                if weight_raw_prob == max_prob:
                    print(f"Sparsity of layer {weight_i} has to be set to 0.")
                    dense_layers.add(weight_i)
        else:
            is_valid = True
    for i in range(len(weights)):
        if i in dense_layers:
            ans.append(0)
        else:
            ans.append(1 - raw_probabilities[i] * epsilon)  
    return ans

class IndexMaskHook:
    def __init__(self, layer:torch.Tensor, scheduler:DST_Scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad:torch.Tensor):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad/self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask

def _create_step_wrapper(scheduler: DST_Scheduler, optimizer: torch.optim.Optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class DST_Scheduler:

    def __init__(
        self, 
        model, 
        optimizer, 
        static_topo=False, 
        sparsity=0, 
        T_end=None, 
        delta:int=100, 
        zeta:float=0.3, 
        random_grow=False, 
        grad_accumulation_n:int=1, 
        ):

        self.model = model
        self.optimizer:torch.optim.Optimizer = optimizer

        self.random_grow = random_grow

        self.W, self._layers_type = get_W(model)

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
            
        self.sparsity = sparsity
        self.N = [torch.numel(w) for w in self.W]

        self.static_topo = static_topo
        self.grad_accumulation_n = grad_accumulation_n
        self.backward_masks:list[torch.Tensor] = None

        self.S = sparse_set(self.W, sparsity, False)

        # sparsify the model
        self.weight_sparsify()

        # scheduler keeps a log of how many times it's called. this is how it does its scheduling
        self.step = 0
        self.dst_steps = 0

        # define the actual schedule
        self.delta_T = delta
        self.zeta = zeta
        self.T_end = T_end

        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects:list[IndexMaskHook] = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_dst_backward_hook', False):
                raise Exception('This model already has been registered to a DST_Scheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_dst_backward_hook', True)

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta

    @torch.no_grad()
    def random_sparsify(self):
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] < 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)
            perm = perm[:s]
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def weight_sparsify(self):
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] < 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            score_drop = torch.abs(w)
            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n-s)
            flat_mask = torch.zeros(n, device=w.device)
            flat_mask[sorted_indices] = 1
            mask = torch.reshape(flat_mask, w.shape)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def reset_momentum(self):#reset the momentum according to the mask
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
                
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            w.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next DST step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end :
            return False

        steps_til_next_dst_step = self.delta_T - (self.step  % self.delta_T)
        return steps_til_next_dst_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.zeta / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step  % self.delta_T) == 0 and self.step  < self.T_end: # check schedule
            self._dst_step()
            self.dst_steps += 1
            return False
        return True

    @torch.no_grad()
    def _dst_step(self):

        total_pruned_num = 0
        total_num = 0

        drop_fraction = self.cosine_annealing()

        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                continue

            current_mask = self.backward_masks[l]

            # calculate raw scores
            score_drop = torch.abs(w)
            if not self.random_grow:
                score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
            else:
                score_grow = torch.rand(self.backward_hook_objects[l].dense_grad.size()).to(device)

            # calculate drop/grow quantities
            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()

            # create drop mask
            sorted_score, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            sorted_score = sorted_score[:n_ones]
            n_prune = int(n_ones * drop_fraction)
            total_num += n_ones
            n_keep = n_ones - n_prune
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)
                
            total_pruned_num += n_prune

            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_prune,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)
            
            new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            # update new weights to be initialized as zeros and update the weight tensors
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            # update the mask
            current_mask.data = mask_combined

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients() 

    @property
    def state_dict(self):
        return {
            'S': self.S,
            'N': self.N,
            'delta_T': self.delta_T,
            'zeta': self.zeta,
            'T_end': self.T_end,
            'static_topo': self.static_topo,
            'grad_accumulation_n': self.grad_accumulation_n,
            'step': self.step,
            'dst_steps': self.dst_steps,
            'backward_masks': self.backward_masks
        }

    def load_state_dict(self, state_dict:Dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

