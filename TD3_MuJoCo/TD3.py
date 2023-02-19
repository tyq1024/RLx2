import copy
import numpy as np
import torch
import torch.nn.functional as F
from TD3_MuJoCo.modules_TD3 import MLPActor, MLPCritic
from DST.DST_Scheduler import DST_Scheduler
from DST.utils import ReplayBuffer, get_W
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
	def __init__(
		self,
		#TD3 basic
		state_dim,
		action_dim,
		max_action,
		hidden_dim=256,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
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
		# RL hyperparameters
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		# Neural networks
		self.actor = MLPActor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = MLPCritic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.total_it = 0
		self.sparse_actor = sparse_actor
		self.sparse_critic = sparse_critic
		self.nstep = nstep
		self.delay_nstep = delay_nstep

		self.current_mean_reward = 0.

		self.writer = SummaryWriter(tb_dir)

		self.tb_interval = int(T_end/10000)

		if self.sparse_actor: # Sparsify the actor at initialization
			self.actor_pruner = DST_Scheduler(model=self.actor, optimizer=self.actor_optimizer, sparsity=actor_sparsity, T_end=int(T_end/self.policy_freq), static_topo=static_actor, sparsity_distribution=sparsity_distribution, **kwargs)
			self.targer_actor_W, _ = get_W(self.actor_target)
		else:
			self.actor_pruner = lambda: True
		if self.sparse_critic: # Sparsify the critic at initialization
			self.critic_pruner = DST_Scheduler(model=self.critic, optimizer=self.critic_optimizer, sparsity=critic_sparsity, T_end=T_end, static_topo=static_critic, sparsity_distribution=sparsity_distribution, **kwargs)
			self.targer_critic_W, _ = get_W(self.critic_target)
		else:
			self.critic_pruner = lambda: True

	def select_action(self, state) -> np.ndarray:
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer:ReplayBuffer, batch_size=256):
		self.total_it += 1

		# Delay to use multi-step TD target
		current_nstep = self.nstep if self.total_it >= self.delay_nstep else 1

		if self.total_it % self.tb_interval == 0: self.writer.add_scalar('current_nstep', current_nstep, self.total_it)
		
		state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(batch_size, current_nstep)
		with torch.no_grad():
			noise = (
				torch.randn_like(action[:,0]) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			accum_reward = torch.zeros(reward[:,0].shape).to(device)
			have_not_done = torch.ones(not_done[:,0].shape).to(device)
			have_not_reset = torch.ones(not_done[:,0].shape).to(device)
			modified_n = torch.zeros(not_done[:,0].shape).to(device)
			for k in range(current_nstep):
				accum_reward += have_not_reset*have_not_done*self.discount**k*reward[:,k]
				have_not_done *= torch.maximum(not_done[:,k], 1-have_not_reset)
				if k == current_nstep - 1:
					break
				have_not_reset *= (1-reset_flag[:,k])
				modified_n += have_not_reset
			modified_n = modified_n.type(torch.long)
			nstep_next_state = next_state[np.arange(batch_size), modified_n[:,0]]
			next_action = (
				self.actor_target(nstep_next_state) + noise
			).clamp(-self.max_action, self.max_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(nstep_next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			if current_nstep == 1:
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
		if self.total_it % self.policy_freq == 0:
			actor_loss:torch.Tensor = -self.critic.Q1(state[:,0], self.actor(state[:,0])).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			if self.actor_pruner():
				self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if self.sparse_critic:
				for w, mask in zip(self.targer_critic_W, self.critic_pruner.backward_masks):
					w.data *= mask

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if self.sparse_actor:
				for w, mask in zip(self.targer_actor_W, self.actor_pruner.backward_masks):
					w.data *= mask
		

		
