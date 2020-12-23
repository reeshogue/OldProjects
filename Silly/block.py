import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ChannelShuffle(nn.Module):
    def forward(self, x):
        y = torch.split(x, 1, dim=1)
        y = random.sample(y, len(y))
        return torch.cat(y, dim=1)

class Rish(nn.Module):
    def __init__(self):
        super(Rish, self).__init__()
        self.param_alpha = nn.Parameter(torch.rand((1,1)))
        self.param_beta = nn.Parameter(torch.rand((1,1)))
        self.eps = 1e-5
    def forward(self, x):
        return self.param_beta * x / torch.clamp((torch.exp(-x) + self.param_alpha), min=self.eps)
    

class Fourier(nn.Module):
    def __init__(self, accuracy=10):
        super(Fourier, self).__init__()
        self.param_alpha = nn.Parameter(torch.rand((1,accuracy)))
        self.param_beta = nn.Parameter(torch.rand((1,accuracy)))
    def forward(self, x):
        y_cos = []
        for i in self.param_alpha[0]:
            y_a = torch.cos(x * i)
            y_cos.append(y_a)
        y_sin = []
        for i in self.param_beta[0]:
            y_b = torch.sin(x * i)
            y_sin.append(y_a)
        y = torch.sum(torch.stack(y_cos, dim=0), dim=0) + torch.sum(torch.stack(y_sin, dim=0), dim=0)
        return y
        
class Attention(nn.Module):
    def __init__(self, size, inchan):
        super(Attention, self).__init__()

        self.query = nn.Conv1d(inchan, inchan, 1)
        self.key = nn.Conv1d(inchan, inchan, 1)
        self.value = nn.Conv1d(inchan, inchan, 1)
    def forward(self, x):
        shape = x.shape
        flat = x.view(shape[0], shape[1], -1)
        
        q = self.query(flat).permute(0, 2, 1)
        k = self.key(flat)
        v = self.value(flat)
        qk = torch.bmm(q, k)
        all_you_need = qk / torch.sum(qk)
        attn = torch.bmm(v, all_you_need)
        attn = attn.view(*shape)
        return attn
        

class ReesBlock(nn.Module):
    def __init__(self, sample_size, inchan, outchan):
        super(ReesBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(inchan, inchan, kernel_size=1)
        self.conv3x3 = nn.Conv2d(inchan, inchan, kernel_size=3)
        self.conv5x5 = nn.Conv2d(inchan, inchan, kernel_size=5)
        self.conv0x0 = nn.Identity()
        self.convolution_param = nn.Parameter(torch.rand(1,1))
         
        self.in1 = nn.InstanceNorm2d(inchan)
        self.norm_param = nn.Parameter(torch.rand(1,1))

        self.in2 = nn.InstanceNorm2d(inchan)
        self.norm_param_2 = nn.Parameter(torch.rand(1,1))

        self.hconv1x1 = nn.Conv2d(inchan, inchan, kernel_size=1)
        self.hconv3x3 = nn.Conv2d(inchan, inchan, kernel_size=3)
        self.hconv5x5 = nn.Conv2d(inchan, inchan, kernel_size=5)
        self.hconv0x0 = nn.Identity()
        self.hidden_convolution_param = nn.Parameter(torch.rand((1, 1)))
        
        self.activation_param = nn.Parameter(torch.rand((1,1)))
        self.activation_param_2 = nn.Parameter(torch.rand((1,1)))

        self.swish = Rish()
        self.swish2 = Rish()
        self.channel_shuffle = ChannelShuffle()
        
        self.attention_param = nn.Parameter(torch.rand((1,1)))
        self.attention = Attention(sample_size, inchan)

        self.outconv = nn.Conv2d(inchan, outchan, 3)
        self.sample = nn.Upsample((sample_size, sample_size))
        self.shortcut_param = nn.Parameter(torch.rand(1,1))
        self.shortcut = nn.Conv2d(inchan, outchan, 3)
        
        self.dropout_param = nn.Parameter(torch.rand((1,1)))
        self.dropout_probability_param = nn.Parameter(torch.rand((1,1)))

    def forward(self, x):
        self.convolution_param.clamp(0, 1)
        if self.convolution_param >= .75:
            y = lambda z: self.conv0x0(z)
        elif self.convolution_param >= .5:
            y = lambda z: self.conv1x1(z)
        elif self.convolution_param >= .25:
            y = lambda z: self.conv3x3(z)
        else:
            y = lambda z: self.conv5x5(z)
        h = y(x)
        h = self.sample(h)

        self.norm_param.clamp(0, 1)
        if self.norm_param >= .5:
            y = lambda z: self.in1(z)
        else:
            y = lambda z: z

        h = y(h)


        self.hidden_convolution_param.clamp(0, 1)
        if self.hidden_convolution_param >= .75:
            y = lambda z: self.hconv0x0(z)
        elif self.hidden_convolution_param >= .5:
            y = lambda z: self.hconv1x1(z)
        elif self.hidden_convolution_param >= .25:
            y = lambda z: self.hconv3x3(z)
        else:
            y = lambda z: self.hconv5x5(z)
        h = y(h)
        
        self.norm_param_2.clamp(0, 1)
        if self.norm_param_2 >= .5:
            y = lambda z: self.in2(z)
        else:
            y = lambda z: z
        h = y(h)

        self.activation_param_2.clamp(0, 1)
        if self.activation_param_2 >= .75:
            y = lambda z: torch.relu(z)
        elif self.activation_param_2 >= .5:
            y = lambda z: self.swish2(z)
        elif self.activation_param_2 >= .25:
            y = lambda z: torch.sigmoid(z)
        h = y(h)
        
        self.attention_param.clamp(0,1)
        if self.attention_param >= .50:
            y = lambda z: self.attention(z)
        else:
            y = lambda z: z

        h = y(h)
        
        h = self.outconv(h)
        h = self.sample(h)
        x = self.shortcut(x)
        x = self.sample(x)
        self.shortcut_param.clamp(0, 1)

        if self.shortcut_param >= .5:
            h = x + h
        else:
            h = h
        
        self.activation_param.clamp(0, 1)
        if self.activation_param >= .75:
            y = lambda z: torch.relu(z)
        elif self.activation_param >= .5:
            y = lambda z: self.swish(z)
        elif self.activation_param >= .25:
            y = lambda z: torch.sigmoid(z)
        else:
            y = lambda z: self.channel_shuffle(z)
        h = y(h)
        
        self.dropout_param.clamp(0,1)
        self.dropout_probability_param.clamp(0,1)
        
        if self.dropout_param >= .50:
            y = lambda z: F.dropout(z, p=self.dropout_probability_param[0].item())
        else:
            y = lambda z: z
        h = y(h)

        return h

class ReesBlock_Lite(nn.Module):
    def __init__(self, sample_size, inchan, outchan, expansion=2):
        super(ReesBlock_Lite, self).__init__()
        self.conv3x3 = nn.Conv2d(inchan, inchan*expansion, kernel_size=3)
        self.conv5x5 = nn.Conv2d(inchan, inchan*expansion, kernel_size=5)
        self.convolution_param = nn.Parameter(torch.rand(1,1))

        self.hconv3x3 = nn.Conv2d(inchan*expansion, outchan, kernel_size=3)
        self.hconv5x5 = nn.Conv2d(inchan*expansion, outchan, kernel_size=5)
        self.hidden_convolution_param = nn.Parameter(torch.rand(1,1))
        
        self.sample = nn.Upsample((sample_size, sample_size))
        self.shortcut = nn.Conv2d(inchan, outchan, 3)

    def forward(self, x):
        self.convolution_param.clamp(0, 1)
        if self.convolution_param >= .5:
            y = lambda z: self.conv3x3(z)
        else:
            y = lambda z: self.conv5x5(z)
        h = y(x)
        h = F.sigmoid(h) * h
        h = self.sample(h)

        self.hidden_convolution_param.clamp(0, 1)
        if self.hidden_convolution_param >= .5:
            y = lambda z: self.hconv3x3(z)
        else:
            y = lambda z: self.hconv5x5(z)
        h = y(h)

        h = self.sample(h)
        x = self.shortcut(x)
        x = self.sample(x)
        h = h + x
        
        return h