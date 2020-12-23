import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import deque
import random
from models import AEPolicy, CriticPolicy, RandomNetworkEncoder


class PorpCab:
	def __init__(self, model):
		self.model =  model
		self.params = list(model.parameters())
		self.param_shape = []
		for param in self.params:
			self.param_shape.append(list(param.cpu().detach().shape))
		length = sum(p.numel() for p in self.model.parameters())

		self.encoder = RandomNetworkEncoder().cpu()
		self.length = length
		self.replay_buffer = deque(maxlen=2000)
		self.count = 0
	def train_critic(self, state, action, next_state, loss):
		self.critic_optim.zero_grad()
		loss = loss.detach().cpu()
		crit = self.critic(torch.cat([state.cpu(), action]))
		loss_zeros = torch.zeros(128)
		loss = loss + loss_zeros
		policy_next_state = self.policy.sample(next_state)[0]
		measure = F.mse_loss(crit, loss+self.critic(torch.cat([next_state, policy_next_state])))
		self.critic_optim.step()
		
	def train_policy(self, state, action, log_prob):
		self.policy_optim.zero_grad()
		crit = (-log_prob) + torch.max(self.critic(torch.cat([state,action]))) 
		self.policy_optim.step()

	def remember(self, state, action, next_state, loss, log_prob):
		self.replay_buffer.append((state, action, next_state, loss, log_prob))
	
	def replay(self, size):
		replay = random.sample(self.replay_buffer, size)
		replay_two = random.sample(self.replay_buffer, size)

		for s, a, n, l, p in replay:
			self.train_critic(s, a, n, l)
			self.train_policy(s, a, p)

		for s, a, n, l, p in replay_two:
			self.train_critic(s, a, n, l)

	def act(self, state):
		return self.policy.sample(state)
	
	def zero_grad(self):
		pass
		
	def step(self, loss, data1, data2, fake_data2):
		data1 = self.encoder(data1.cpu())
		data2 = self.encoder(data2.cpu())
		fake_data2 = self.encoder(fake_data2.cpu())

		state = torch.cat([data1.view(-1), data2.view(-1), fake_data2.view(-1)])
		if self.count == 0:
			self.policy = AEPolicy(state.shape[0], self.length).cpu()

			self.policy_optim = torch.optim.AdamW(self.policy.parameters(), lr=.0004, amsgrad=True)
			self.critic = CriticPolicy(state.shape[0]+self.length).cpu()
			self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=.0004, amsgrad=True)


		action,log_prob,mean = self.act(state)

		if self.count == 0:
			self.prev_action = action
			self.prev_state = state
			

		self.remember(self.prev_state, self.prev_action, state, loss, log_prob)
		if torch.rand(1) < .5:
			self.replay(1)

		self.reshaped = []
		acc = 0
		prev_shapes = 0
		for layer in self.param_shape:
			if isinstance(layer, float):
				layer=[int(layer)]
			shapes = 1
			for i in layer:
				shapes *= i
			shapes = np.int(shapes) 
			
			self.reshaped.append(torch.reshape(action.detach()[prev_shapes:shapes+prev_shapes], tuple(layer)))
			prev_shapes = shapes + prev_shapes


		for p, i in zip(self.model.parameters(), self.reshaped):
			try:
				p.copy_(i)
			except RuntimeError:
				p.data = i
		self.prev_action = action
		self.prev_state = state
		self.count += 1