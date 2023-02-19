import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TD3_MuJoCo.modules_TD3 import MLPActor, MLPCritic
from DST.DST_Scheduler import DST_Scheduler, sparse_set
from DST.utils import ReplayBuffer, get_W
from DST.utils import show_sparsity
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from modules_SAC import DiagGaussianActor, DoubleQCritic
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(object):
    def __init__(
        self,
        #SAC basic
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        discount=0.99,
        lr=3e-4,
        actor_update_frequency=1, 
        tau=0.005, 
        critic_target_update_frequency=1,
        log_std_bounds=None,
        #sparse setting
        sparse_actor=False,
        sparse_critic=False,
		static_actor=False,
		static_critic=False,
		actor_sparsity=0,
		critic_sparsity=0,
		sparsity_distribution=None,
		#DST setting
		T_end=975000,
		#test
		nstep=1,
		delay_nstep=0,
		#
		tb_dir=None,
		#args2Pruner
		**kwargs
    ):

        self.critic = DoubleQCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(state_dim, action_dim, max_action, hidden_dim, log_std_bounds).to(device)

        self.log_alpha = torch.tensor(0.0).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=lr)

        self.discount = discount
        self.tau =tau
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.total_it = 0
        self.sparse_actor = sparse_actor
        self.sparse_critic = sparse_critic
        self.nstep = nstep
        self.delay_nstep = delay_nstep
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency

        self.current_mean_reward = 0.

        self.writer = SummaryWriter(tb_dir)

        self.tb_interval = int(T_end/10000)

        if self.sparse_actor:
            self.actor_pruner = DST_Scheduler(model=self.actor, optimizer=self.actor_optimizer, sparsity=actor_sparsity, T_end=int(T_end/self.actor_update_frequency), static_topo=static_actor, sparsity_distribution=sparsity_distribution, **kwargs)
            self.supervised_actor_list = {'mask_explore':[], 'mask_change':[], 'mask_eq_size':[], 'mask_eq_sp':[]}
            self.sup_mask_actor = copy.deepcopy(self.actor_pruner.backward_masks)
            self.last_mask_actor = copy.deepcopy(self.actor_pruner.backward_masks)
        else:
            self.actor_pruner = lambda: True
        if self.sparse_critic:
            self.critic_pruner = DST_Scheduler(model=self.critic, optimizer=self.critic_optimizer, sparsity=critic_sparsity, T_end=T_end, static_topo=static_critic, sparsity_distribution=sparsity_distribution, **kwargs)
            self.targer_critic_W, _ = get_W(self.critic_target)
            self.supervised_critic_list = {'mask_explore':[], 'mask_change':[], 'mask_eq_size':[], 'mask_eq_sp':[]}
            self.sup_mask_critic = copy.deepcopy(self.critic_pruner.backward_masks)
            self.last_mask_critic = copy.deepcopy(self.critic_pruner.backward_masks)
        else:
            self.critic_pruner = lambda: True

    def select_action(self, state, sample=False) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.cpu().data.numpy().flatten(), dist.mean.cpu().data.numpy().flatten()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, replay_buffer:ReplayBuffer, batch_size=256):
        self.total_it += 1

        current_n = self.nstep if self.total_it >= self.delay_nstep else 1

        if self.total_it % self.tb_interval == 0: self.writer.add_scalar('current_nstep', current_n, self.total_it)
            
        state, action, next_state, reward, not_done, _ , reset_flag= replay_buffer.sample(batch_size, current_n)

        with torch.no_grad():
            accum_reward = torch.zeros(reward[:,0].shape).to(device)
            have_not_done = torch.ones(not_done[:,0].shape).to(device)
            have_not_reset = torch.ones(not_done[:,0].shape).to(device)
            modified_n = torch.zeros(not_done[:,0].shape).to(device)
            nstep_next_action = torch.zeros(action[:,0].shape).to(device)
            for k in range(current_n):
                accum_reward += have_not_reset*have_not_done*self.discount**k*reward[:,k]
                have_not_done *= torch.maximum(not_done[:,k], 1-have_not_reset)
                dist = self.actor(next_state[:,k])
                next_action = dist.rsample()
                nstep_next_action += have_not_reset*have_not_done*(next_action-nstep_next_action)
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
                accum_reward += have_not_reset*have_not_done*self.discount**(k+1)*(- self.alpha.detach() * log_prob)
                if k == current_n - 1:
                    break
                have_not_reset *= (1-reset_flag[:,k])
                modified_n += have_not_reset
            modified_n = modified_n.type(torch.long)
            nstep_next_state = next_state[np.arange(batch_size), modified_n[:,0]]
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(nstep_next_state, nstep_next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if current_n == 1:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount * target_Q
            else:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount**(modified_n + 1) * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state[:,0], action[:,0])
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.total_it % self.tb_interval == 0: self.writer.add_scalar('critic_loss',critic_loss.item(), self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.critic_pruner():
            self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.actor_update_frequency == 0:
            dist = self.actor(state[:,0])
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.critic(state[:,0], action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.actor_pruner():
                self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            if self.total_it % self.tb_interval == 0: self.writer.add_scalar('alpha',self.alpha.item(), self.total_it)

            # Update the frozen target models
        if self.total_it % self.critic_target_update_frequency == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if self.sparse_critic:
                for w, mask in zip(self.targer_critic_W, self.critic_pruner.backward_masks):
                    w.data *= mask
        
