import torch
import torch.nn as nn
 
class Block(nn.Module):
    def __init__(self, sample_size, inchan, outchan, kernel_size=3):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(inchan, inchan, kernel_size)
        self.conv2 = nn.Conv2d(inchan, outchan, kernel_size)
        self.conv_shortcut = nn.Conv2d(inchan, outchan, kernel_size)
        self.sample = nn.Upsample((sample_size, sample_size))
        self.conv_b = nn.Conv2d(outchan, outchan, kernel_size)
        self.conv_a = nn.Conv2d(outchan, outchan, kernel_size)
        
        self.activation = nn.PReLU(outchan)
        self.activation_b = nn.PReLU(outchan)

        self.rand_param = nn.Parameter(torch.ones((1,1)))
        self.forget_param = nn.Parameter(torch.ones((1,1)))
        self.forget_param_y = nn.Parameter(torch.ones((1,1)))
        
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv_shortcut.weight)
        nn.init.xavier_uniform_(self.conv_b.weight)
        nn.init.xavier_uniform_(self.conv_a.weight)

        nn.init.constant_(self.conv.bias, .00787499699)
        nn.init.constant_(self.conv2.bias, .00787499699)
        nn.init.constant_(self.conv_shortcut.bias, .00787499699)
        nn.init.constant_(self.conv_b.bias, .00787499699)
        nn.init.constant_(self.conv_a.bias, .00787499699)

    def forward(self, x, noise=True):
        y = self.sample(self.conv2(self.conv(self.sample(x))))
        y = self.activation(y)
        x = self.sample(self.conv_shortcut(self.sample(x)))
        if noise:
            y_a = self.sample(self.conv_a(y))
            y_b = self.sample(self.conv_b(y))
            y_noise = (y_b * torch.randn(y_b.shape)) + y_a
            y = y + (self.rand_param * y_noise)
        param_y_forg, param_forg = torch.softmax(torch.cat([self.forget_param_y, self.forget_param]), dim=1)
        y = (param_y_forg * y) + (param_forg * x)
        y = self.activation_b(y)
        return y

class DesireMapper(nn.Module):
    def __init__(self):
        super(DesireMapper, self).__init__()
        self.linear = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 64*64)
    def forward(self, dr, dt):
        tensor = torch.cat([dr, dt], dim=1)
        return torch.reshape(self.linear2(self.linear(tensor)), (1,1,64,64))

class Brain(nn.Module):
    def __init__(self, action_size=100, critic_bool=False, depth=7, goal_size=32):
        super(Brain, self).__init__()
        self.mapper = DesireMapper()
        self.conv1 = Block(128, 6, 11)
        self.conv1_2 = Block(64, 11, 11)

        self.intersection_conv = Block(64, 12, 12)
        self.conv2 = nn.ModuleList([Block(32, 12, 12) for i in range(depth)])

        self.conv3_goal = Block(32, 12, 1)
        self.conv3 = Block(action_size, 12, 3)
        
        self.goal_flatten = nn.Flatten()
        self.linear_goals = nn.Linear(32*32, goal_size)

        self.policy_conv = nn.ModuleList([Block(action_size, 3, 3) for i in range(depth)])
        self.policy_head = Block(action_size, 3, 3)
    def forward(self, x, desired_reward=None, desired_time=None):        
        map = self.mapper(desired_reward, desired_time)
        y_merge = self.conv1(x)
        y_merge = self.conv1_2(y_merge)
        y_merge = torch.cat([y_merge, map], dim=1)
        y_merge = self.intersection_conv(y_merge)

        for h in self.conv2:
            y_merge = h(y_merge)
        y_goal = self.linear_goals(self.goal_flatten(self.conv3_goal(y_merge)))
        y_merge = self.conv3(y_merge)
         
        for h in self.policy_conv:
            y_merge = h(y_merge)
        action = self.policy_head(y_merge)
        return action, y_goal

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = Block(64, 9, 6)
        self.conv2 = Block(12, 6, 6)
        self.conv3 = Block(6, 6, 6)
        self.conv4 = Block(6, 6, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(6*6, 1)
    def forward(self, x):
        
        y = self.conv(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.flatten(y)
        return self.linear(y)