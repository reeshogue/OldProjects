import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import deque
import random

class AEPolicy(nn.Module):
	def __init__(self, params_size):
		super(AEPolicy, self).__init__()
		self.linear = nn.Linear(params_size, 256)
		self.hidden = nn.ModuleList([nn.Linear(256, 256) for i in range(5)])
		self.linear2 = nn.Linear(256, params_size)
	def forward(self, x):
		x = self.linear(x)
		for i in self.hidden:
			x = i(x)
		return self.linear2(x)

class CriticPolicy(nn.Module):
	def __init__(self, params_size):
		super(CriticPolicy, self).__init__()
		self.linear = nn.Linear(params_size, 256)
		self.hidden = nn.ModuleList([nn.Linear(256, 256) for i in range(5)])
		self.linear2 = nn.Linear(256, 128)
	def forward(self, x):
		x = self.linear(x)
		for i in self.hidden:
			x = i(x)
		return self.linear2(x)

class PorpCab:
	def __init__(self, model):
		self.model =  model
		self.params = list(model.parameters())
		self.param_shape = []
		for param in self.params:
			self.param_shape.append(list(param.cpu().detach().shape))
		length = sum(p.numel() for p in self.model.parameters())

		self.policy = AEPolicy(int(length)).cpu()

		self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
		self.critic = CriticPolicy(int(length)).cpu()
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
		self.replay_buffer = deque(maxlen=200)
	def train_critic(self, loss, future_params, params):
		loss = loss.detach().cpu()
		self.critic_optim.zero_grad()
		crit = self.critic(params.cpu())
		loss_zeros = torch.zeros(128)
		loss = loss + loss_zeros
		measure = F.mse_loss(crit, loss+self.critic(future_params).detach())
		self.critic_optim.step()
		
	def train_policy(self, params):
		self.policy_optim.zero_grad()
		policy_params = self.policy(params)
		crit = torch.max(self.critic(policy_params))
		self.policy_optim.step()
		
	def remember(self, params, next_params, loss):
		self.replay_buffer.append((params, next_params, loss))
	
	def replay(self, size):
		replay = random.sample(self.replay_buffer, size)
		replay_two = random.sample(self.replay_buffer, size)

		for p, n, l in replay:
			self.train_critic(l, n, p)
			self.train_policy(p)

		for p, n, l in replay_two:
			self.train_critic(l, n, p)
	def act(self, current_params):
		return self.policy(current_params).detach()

	def step(self, loss):
		self.params = torch.cat([param.view(-1) for param in self.model.parameters()]).cpu()
		action_new_params = self.act(self.params.cpu())
		self.remember(self.params, action_new_params, loss)
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
			
			self.reshaped.append(torch.reshape(action_new_params.detach()[prev_shapes:shapes+prev_shapes], tuple(layer)))
			prev_shapes = shapes + prev_shapes


		for p, i in zip(self.model.parameters(), self.reshaped):
			try:	
				p.copy_(i)
			except RuntimeError:
				p.data = i

# class MetaBackProp(torch.optim.Optimizer):
# 	def __init__(self, params):
		
# 		self.param_shape_list = np.array([])
# 		for param in list(params):
# 			np.append(self.param_shape_list, list(param.size()))

# 		pseudo_lr = 1e-4
# 		pseudo_defaults = dict(lr=pseudo_lr)
		
# 		self.policy = AEPolicy(length)
# 		self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=pseudo_lr)
# 		super(MetaBackProp, self).__init__(params, pseudo_defaults)

# 	def step(self, closure=None):
# 		params = torch.cat([p.view(-1) for p in self.param_groups])
# 		self.policy_optim.zero_grad()
# 		quit()




# class MetaLR:
# 	def __init__(self, model, optimizer, device, model2=None):
# 		self.model = model
# 		self.model2 = model2
# 		params = self.get_params()
# 		self.actor = NeuralPolicy(len(params)).to(device)
# 		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
# 		self.optim = optimizer
# 		self.device = device
# 		self.replay_buffer = deque(maxlen=16)
# 
# 	def get_params(self):
# 		params1 = torch.cat([param.view(-1) for param in self.model.parameters()])
# 		if self.model2:
# 			params2 = torch.cat([param.view(-1) for param in self.model2.parameters()])
# 
# 		params = torch.cat([params1, params2]) if self.model2 else params1
# 		return params
# 
# 	def step(self, loss):
# 		self.actor.zero_grad()
# 		params = self.get_params()
# 		a = self.actor(params)
# 		dist = torch.distributions.Bernoulli(a)
# 		loss = dist.log_prob(a) * loss
# 		self.actor_optim.step()
# 
# 		a = self.actor(params)
# 		action_lr = torch.mean(a).item()
# 		if action_lr < 0.0:
# 			action_lr = 0.00000000000001
# 		elif action_lr > .6:
# 			action_lr = 0.0000000000001
# 		print("LR of", self.model.name, "is now", action_lr)
# 		self.optim.param_groups[0]['lr'] = action_lr