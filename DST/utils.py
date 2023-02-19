from __future__ import annotations
from logging import raiseExceptions
from typing import Tuple
import torch
import numpy as np

def show_sparsity(state_dict, only_weight=True, to_print=True):
	total_zero=0
	total_dense=0
	for name,layer in state_dict.items():
		if only_weight and 'weight' not in name:
			continue
		layer=layer.eq(0)
		dense=layer.numel()
		total_dense+=dense
		layer=layer.detach().cpu().numpy()
		zero=np.sum(layer.ravel())
		total_zero+=zero
		if zero != 0 and to_print: print(f"{name} : {zero/dense}")
	if to_print: print(f"whole model: {total_zero/total_dense}")
	return total_zero/total_dense

def get_weighted_layers(model, i=0, layers=None, layers_type=None):
    if layers is None:
        layers = []
    if layers_type is None:
        layers_type = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for _, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            layers_type.append('L')
        elif hasattr(p, 'weight'):
            layers.append([p])
            layers_type.append('C')
        else:
            _, layers_type, i = get_weighted_layers(p, i=i + 1, layers=layers, layers_type=layers_type)

    return layers, layers_type, i 

def get_W(model) -> Tuple[list[torch.Tensor], list[str]]:
    layers, layers_type, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)

    assert len(W) == len(layers_type)

    return W, layers_type

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.left_ptr = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.action_mean = np.zeros((max_size, action_dim))
        self.reset_flag = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, action_mean, reset_flag):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.action_mean[self.ptr] = action_mean
        self.reset_flag[self.ptr] = float(reset_flag)

        self.ptr = (self.ptr + 1) % self.max_size
        if self.left_ptr == self.ptr:
            self.left_ptr = (self.left_ptr + 1) % self.max_size

    def sample(self, batch_size, nstep=1):
        ind = (np.random.randint(0, self.size-nstep+1, (batch_size,1)) + self.left_ptr + np.arange(nstep))%self.max_size
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.action_mean[ind]).to(self.device),
            torch.FloatTensor(self.reset_flag[ind]).to(self.device)
        )

    @property
    def size(self):
        return (self.ptr+self.max_size-self.left_ptr)%self.max_size

    def shrink(self):
        drop_num = int(0.1*self.size)
        self.left_ptr = (self.left_ptr + drop_num) % self.max_size