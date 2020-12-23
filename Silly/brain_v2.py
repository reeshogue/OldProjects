from block import ReesBlock_Lite as ReesBlock
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
        self.param_alpha = nn.Parameter(torch.ones((1,1)))
        self.param_beta = nn.Parameter(torch.ones((1,1)))
    def forward(self, x):
        return self.param_beta * x / (torch.exp(-x) + self.param_alpha)

class DesireMapper(nn.Module):
    def __init__(self):
        super(DesireMapper, self).__init__()
        self.linear = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 200*200)
    def forward(self, dr, dt):
        tensor = torch.cat([dr, dt], dim=1)
        return torch.reshape(self.linear2(self.linear(tensor)), (1,1,200,200))

class Brain(nn.Module):
    def __init__(self, action_size=100):
        super(Brain, self).__init__()
        self.mapper = DesireMapper()
        self.conv1 = ReesBlock(action_size, 4, 3)
        self.conv5 = ReesBlock(action_size, 3, 3)
        self.policy_head = ReesBlock(action_size, 3, 3)

    def forward(self, x, desired_reward, desired_time):
        orig_x = x
        map = self.mapper(desired_reward, desired_time)
        x = torch.cat([x, map], 1)
        y = self.conv1(x)
        y = self.conv5(y)
        image_action = self.policy_head(y) + orig_x
        return image_action

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = ReesBlock(64, 3, 8)
        self.conv2 = ReesBlock(12, 8, 8)
        self.conv3 = ReesBlock(8, 8, 8)
        self.conv4 = ReesBlock(8, 8, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8*8, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
    def forward(self, x):
        y = self.conv(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.flatten(y)
        return torch.sigmoid(self.linear(y))
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_false = self.forward(x)
        loss = F.mse_loss(y_false, y)
        loss.backward()
        self.optimizer.step()        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = ReesBlock(64, 6, 8)
        self.conv2 = ReesBlock(12, 8, 8)
        self.conv3 = ReesBlock(8, 8, 8)
        self.conv4 = ReesBlock(8, 8, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8*8, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
    def forward(self, x):
        y = self.conv(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.flatten(y)
        return self.linear(y)
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_false = self.forward(x)
        loss = F.mse_loss(y_false, y)
        loss.backward()
        self.optimizer.step()        


class Environment(nn.Module):
    def __init__(self, action_size=200):
        super(Environment, self).__init__()
        self.conv1 = ReesBlock(action_size, 9, 6)
        self.conv2 = ReesBlock(action_size, 6, 6)
        self.conv3 = ReesBlock(action_size, 6, 6)
        self.conv3_goal = ReesBlock(32, 6, 1)
        self.goal_flatten = nn.Flatten()
        self.linear_goals = nn.Linear(32*32, 1)

        self.policy_head = ReesBlock(action_size, 6, 6)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

    def forward(self, x):
        y_merge = self.conv1(x)
        y_merge = self.conv2(y_merge)
        y_merge = self.conv3(y_merge)
        y_goal = self.linear_goals(self.goal_flatten(self.conv3_goal(y_merge)))
        action = self.policy_head(y_merge)        
        action = torch.sigmoid(action)
        y_goal = torch.softmax(y_goal, -1)
        return action, y_goal
    def train_step(self, pair, ns, reward):
        self.optimizer.zero_grad()
        y_false_state, y_false_reward = self.forward(pair)
        loss_a = F.mse_loss(y_false_state, ns)
        loss_b = F.mse_loss(y_false_state, ns)
        loss = loss_a + loss_b
        loss.backward()
        self.optimizer.step()

class InverseDynamics(nn.Module):
    def __init__(self, action_size=200):
        super(InverseDynamics, self).__init__()
        self.conv1 = ReesBlock(action_size, 6, 3)
        self.conv2 = ReesBlock(action_size, 6, 3)
        self.cat = lambda z: torch.cat(z, 1)
        self.action_out = ReesBlock(action_size, 6, 6)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
    def forward(self, x):
        state, next = x
        s1, s2 = self.conv1(state), self.conv2(next)
        y = self.cat([s1, s2])
        y = self.action_out(y)
        return y, self.conv1 
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_false_action = self.forward(x)[0]
        loss_a = F.mse_loss(y_false_action, y)
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    x = torch.randn((1, 20, 20, 20))
    layer = ReesBlock_v2(20, 20, 3)
    y = layer(x)