import torch.nn as nn
import torch.nn.functional as F
import torch
import time 

class HelpfulSeperableConv2d(nn.Module):
	def __init__(self, inchan, outchan, kernel, resdrop, stride=1, activation=None, downsample=True):
		super(HelpfulSeperableConv2d, self).__init__()
		self.conv = nn.Conv2d(inchan, inchan, kernel, groups=inchan)
		self.pointwise = nn.Conv2d(inchan, outchan, 1, 1)
		self.downsample = nn.Upsample((resdrop, resdrop)) if downsample else nn.Identity()
		self.activation = nn.PReLU(outchan) if activation is None else activation
		self.normalization = nn.InstanceNorm2d(outchan)
	def forward(self, x):
		init_x = x
		x = self.conv(x)
		x = self.pointwise(x)
		x = self.normalization(x)
		x = self.downsample(x)
		x = self.activation(x)
		return x

class Lambda(nn.Module):
	def __init__(self, lamb):
		super(Lambda, self).__init__()
		self.lamb = lamb
	def forward(self, x):
		return self.lamb(x)

class ReesBlock(nn.Module):
	def __init__(self, inchan, outchan, kernel, resdrop, activation=None, downsample=True, norm=3, attn = False):
		super(ReesBlock, self).__init__()
		self.norm = norm
		self.attn = attn
		self.conv = nn.Conv2d(inchan, outchan, kernel)
		self.conv2 = nn.Conv2d(inchan, outchan, kernel)
		self.downsample = nn.Upsample((resdrop, resdrop))
		self.activation = nn.PReLU(outchan) if activation is None else activation
		self.instance_normalization = nn.ModuleList([nn.InstanceNorm2d(inchan) for i in range(norm)])
		self.instance_normalization_params = nn.ParameterList([nn.Parameter(torch.rand(1)) for i in range(norm)])
		self.batch_normalization = nn.ModuleList([nn.BatchNorm2d(outchan) for i in range(norm)])
		self.batch_normalization_params = nn.ParameterList([nn.Parameter(torch.rand(1)) for i in range(norm)])
		self.layer_normalization = nn.ModuleList([nn.LayerNorm((resdrop, resdrop)) for i in range(norm)])
		self.layer_normalization_params = nn.ParameterList([nn.Parameter(torch.rand(1)) for i in range(norm)])
		self.param1 = nn.Parameter(torch.rand(1))
		self.param2 = nn.Parameter(torch.rand(1))

		self.query = nn.Conv2d(outchan, outchan, 1)
		self.key = nn.Conv2d(outchan, outchan, 1)
		self.value = nn.Conv2d(outchan, outchan, 1)
	def forward(self, x):
		real_x = x
		x = self.conv(x)
		x = self.downsample(x)
		init_x = x
		for i in range(self.norm):
			x = x + (self.instance_normalization[i](init_x) * self.instance_normalization_params[i])
			x = x + (self.layer_normalization[i](init_x) * self.layer_normalization_params[i])
			x = x + (self.batch_normalization[i](init_x) * self.batch_normalization_params[i])
		x = x * self.param1
		x = x + (self.param2 * self.downsample(self.conv2(real_x)))
		
		if self.attn:
			dims = (x.shape[0], -1, x.shape[2] * x.shape[3])
			query = self.query(x).view(dims)
			key = self.key(x).view(dims).permute(0, 2, 1)
			value = self.value(x).view(dims)
			
			attn = torch.softmax(torch.bmm(key, query), dim=-1)

			value = torch.bmm(value, attn).view(x.shape)
			x = value + x
		return x		

class AEDiscriminator(nn.Module):
	def __init__(self):
		super(AEDiscriminator, self).__init__()
		self.conv2 = ReesBlock(3, 6, 5, 32, downsample=True, attn=True)

		self.conv6 = ReesBlock(6, 3, 5, 256)
		self.conv7 = ReesBlock(3, 3, 5, 256, downsample=True)
	def forward(self, x):
		y = self.conv7(self.conv6(self.conv2(x)))
		return F.mse_loss(x, y)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv2 = ReesBlock(3, 6, 5, 32, downsample=True)
		self.conv6 = ReesBlock(6, 6, 5, 16)
		self.conv7 = ReesBlock(3, 1, 5, 16, downsample=True, activation=torch.tanh)
	def forward(self, x):
		y = self.conv7(self.conv6(self.conv2(x)))
		return y	

class Generator(nn.Module):
	def __init__(self, noise=True):
		super(Generator, self).__init__()
		self.list_res = [256,  128,  64, 32, 16]
		self.encoders = nn.ModuleList([])
		self.decoders = nn.ModuleList([])
		self.noise = noise
		if self.noise:
			self.noise_encoder = nn.ModuleList([])
			self.noise_encoder.append(nn.Linear(512, 16*16*3))
			self.noise_encoder.append(Lambda(lambda x: x.view(x.shape[0], 3, 16, 16)))
			self.noise_encoder.append(ReesBlock(3, 3, 5, 32))
			self.noise_encoder.append(ReesBlock(3, 3, 5, 64))
						
		for i in self.list_res:
			if i < 32:
				self.encoders.append(ReesBlock(3,3,5,i, attn=True))
			else:
				self.encoders.append(ReesBlock(3,3,5,i))
		for i in reversed(self.list_res):
			if i < 64:
				self.encoders.append(ReesBlock(3,3,5,i, attn=True))
			else:
				self.encoders.append(ReesBlock(3,3,5,i))
	
	def forward(self, x, noise=True):
		if noise:
			noise = torch.randn(1, 512)
			for i in self.noise_encoder:
				noise = i(noise)
		a = x
		z = []

		for i in range(len(self.encoders)):
			a = self.encoders[i](a)
			if i == 2:
				a = a + noise
		for i in self.decoders:
			a = i(a)
		
		return torch.tanh(a)

class OriginalGenerator(nn.Module):
	def __init__(self):
		super(OriginalGenerator, self).__init__()
		self.depth = 8
		self.block1 = ReesBlock(1, 64, 5, 32)
		self.block2 = ReesBlock(64, 32, 5, 64)
		self.block3 = ReesBlock(32, 16, 5, 128)
		self.block4 = ReesBlock(16, 3, 5, 256)
		self.block5 = ReesBlock(3, 3, 5, 256)
		self.block6 = ReesBlock(3, 3, 5, 256)

		self.name = "generator"
	
	def forward(self, x, hint=None, training=True):
		a = x
		z = []

		a = self.block1(a)
		a = self.block2(a)
		a = self.block3(a)
		a = self.block4(a)
		a = self.block5(a)
		a = self.block6(a)
		return torch.tanh(a)

class Real(nn.Module):
	def __init__(self):
		super(Real, self).__init__()
		self.real_base = -torch.ones(1, 1, 16, 16) +  (torch.rand(1, 1, 16, 16) * .3)
	def forward(self):
		return self.real_base

class Fake(nn.Module):
	def __init__(self):
		super(Fake, self).__init__()
		self.fake_base = torch.ones(1, 1, 16, 16) - (torch.rand(1, 1, 16, 16) * 0.3)
	def forward(self):
		return self.fake_base
