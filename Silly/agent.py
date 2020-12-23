import torch
import numpy as np
from brain_v2 import Brain, Critic, Environment, InverseDynamics
from collections import deque
from scipy.special import softmax
import sys

'''Upside Down RL Agent.'''

class UDRLAgent:
    def __init__(self, action_size=200, mcs=1, goal_size=32):
        self.actor = Brain(action_size=action_size, goal_size=goal_size)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.timestep = 0
        self.maxlen = 300
        self.memory = deque(maxlen=self.maxlen)
        self.reward_memory = deque(maxlen=self.maxlen)
        
        self.desired_reward = torch.Tensor([[10]])
        self.desired_time = torch.Tensor([[10]])


    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, self.timestep, next_state))
        next_action = self.actor(next_state, self.desired_reward, self.desired_time+1)[0]
        self.reward_memory.append(reward)
    def act(self, state):
        desired_time = self.maxlen - self.timestep

        if desired_time == 0:
            self.timestep = 0

        mean, std = np.mean(self.reward_memory), np.std(self.reward_memory)
        desired_reward = np.random.uniform(mean, (mean+std))
        self.desired_reward = torch.Tensor([[desired_reward]])
        
        self.desired_time = torch.Tensor([[desired_time]])
        action, action_goals = self.actor(state, self.desired_reward.cuda(), self.desired_time.cuda())

        self.timestep += 1

        return [action.detach(), action_goals.detach()]

    def train(self, state, action, reward):
        reward = reward
        memory = np.random.choice(len(self.memory), p=softmax(np.array(self.reward_memory)))
        memory = [self.memory[memory]]
        
        for state_random, action_random, reward_random, timestep_random, next_state_random in memory:
            action_random, goal_action_random = action_random

            action_random = action_random.detach()
            goal_action_random = goal_action_random.detach()
            d_h = self.timestep - timestep_random
            d_r = reward_random + reward
            d_r = torch.Tensor([[d_r]])
            d_h = torch.Tensor([[d_h]])

            policy_action, goal_action = self.actor(state_random, d_r, d_h)
            loss_g = torch.nn.functional.mse_loss(goal_action, goal_action_random)
            loss_a = torch.nn.functional.mse_loss(policy_action, action_random)
            loss_a.backward(retain_graph=True)
            loss_g.backward()
            loss = loss_a + loss_g
            self.actor_optim.step()
            return loss

'''Upside Down Dyna-Q Agent.'''

class UDQLMAgent:
    def __init__(self, action_size=200, mcs=1, goal_size=32):
        self.actor = Brain(action_size=action_size, goal_size=goal_size)
        self.environmemt = Environment()
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.timestep = 0
        self.maxlen = 3000
        self.simulation_num = 3
        self.memory = deque(maxlen=self.maxlen)
        self.reward_memory = deque(maxlen=self.maxlen)

        self.critic = Critic()
        self.desired_reward = torch.Tensor([[10]])
        self.desired_time = torch.Tensor([[10]])


    def remember(self, state, action, reward, next_state):
        next_action = self.actor(next_state, self.desired_reward, self.desired_time+1)
        critique = self.critic(torch.cat([next_state, next_action[0]], dim=1))
        self.memory.append((state, action, reward, self.timestep, next_state, critique))

        self.reward_memory.append(critique.detach().item())
    def act(self, state):
        desired_time = self.maxlen - self.timestep

        if desired_time == 0:
            self.timestep = 0

        mean, std = np.mean(self.reward_memory), np.std(self.reward_memory)
        desired_reward = np.random.uniform(mean, (mean+std))
        self.desired_reward = torch.Tensor([[desired_reward]])
        
        self.desired_time = torch.Tensor([[desired_time]])
        action, action_goals = self.actor(state, self.desired_reward, self.desired_time)

        self.timestep += 1

        return [action.detach(), action_goals.detach()]

    def train(self, state, action, reward):
        reward = reward
        memory = np.random.choice(len(self.memory), p=softmax(np.array(self.reward_memory)))
        memory = [self.memory[memory]]
        
        for state_random, action_random, reward_random, timestep_random, next_state_random, critic_random in memory:
            action_random, goal_action_random = action_random

            action_random = action_random.detach()
            state_action_pair_random = torch.cat([state_random, action_random], dim=1)
            state_action_pair = torch.cat([state, action[0]], dim=1)

            d_Q = self.critic(state_action_pair).detach() + critic_random.detach()
            d_h = self.timestep - timestep_random
            d_h = torch.Tensor([[d_h]])


            next_state_random_action_pair = torch.cat([next_state_random, self.actor(next_state_random, d_Q, torch.Tensor([[self.timestep-timestep_random]]))[0].detach()], dim=1)
            goal_action_random = goal_action_random.detach()

            Q_next = self.critic(next_state_random_action_pair).detach()
            self.critic.train_step(state_action_pair_random, reward_random + 0.96 * Q_next)
            
            self.environmemt.train_step(state_action_pair_random, next_state_random, reward_random)

            simstep = timestep_random
            
            #Simulation training of Q,
            for i in range(self.simulation_num):
                actions_rand = self.actor(next_state_random, d_Q, torch.Tensor([[self.timestep - simstep]]))[0].detach()
                state_action_pair = torch.cat([next_state_random, actions_rand], dim=1)
                state_next_random, reward_next_random = self.environmemt(state_action_pair)
                actions_rand_2 = self.actor(next_state_random, d_Q, torch.Tensor([[self.timestep - (simstep-1)]]))[0].detach()
                
                Q_next = self.critic(torch.cat([state_next_random, actions_rand_2], 1)).detach()
                self.critic.train_step(state_action_pair, reward_next_random + (0.96 * Q_next))
                next_state_random = state_next_random.detach()
                simstep += 1


            policy_action, goal_action = self.actor(state_random, d_Q, d_h)
            loss_g = torch.nn.functional.mse_loss(goal_action, goal_action_random)
            loss_a = torch.nn.functional.mse_loss(policy_action, action_random)
            loss_a.backward(retain_graph=True)
            loss_g.backward()
            loss = loss_a + loss_g
            self.actor_optim.step()
            return loss

'''Upside Down Q-Learning'''

class UDQLAgent:
    def __init__(self, action_size=200, mcs=1, goal_size=32):
        self.actor = Brain(action_size=action_size)
        self.environmemt = Environment()
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.timestep = 0
        self.maxlen = 3000
        self.simulation_num = 3
        self.memory = deque(maxlen=self.maxlen)
        self.reward_memory = deque(maxlen=self.maxlen)

        self.critic = Critic()
        self.desired_reward = torch.Tensor([[10]])
        self.desired_time = torch.Tensor([[10]])


    def remember(self, state, action, reward, next_state):
        next_action = self.actor(next_state, self.desired_reward, self.desired_time+1)
        critique = self.critic(torch.cat([next_state, next_action[0]], dim=1))
        self.memory.append((state, action, reward, self.timestep, next_state, critique))

        self.reward_memory.append(critique.detach().item())
    def act(self, state):
        desired_time = self.maxlen - self.timestep

        if desired_time == 0:
            self.timestep = 0

        mean, std = np.mean(self.reward_memory), np.std(self.reward_memory)
        desired_reward = np.random.uniform(mean, (mean+std))
        self.desired_reward = torch.Tensor([[desired_reward]]).cuda()
        
        self.desired_time = torch.Tensor([[desired_time]]).cuda()
        action, action_goals = self.actor(state, self.desired_reward, self.desired_time)

        self.timestep += 1

        return [action.detach(), action_goals.detach()]

    def train(self, state, action, reward):
        reward = reward
        memory = np.random.choice(len(self.memory), p=softmax(np.array(self.reward_memory)))
        memory = [self.memory[memory]]
        
        for state_random, action_random, reward_random, timestep_random, next_state_random, critic_random in memory:
            action_random, goal_action_random = action_random
            
            #For d_Q calculations. -- Require random state action and latest.
            action_random = action_random.detach()
            state_action_pair_random = torch.cat([state_random, action_random], dim=1)
            state_action_pair = torch.cat([state, action[0]], dim=1)

            d_Q = self.critic(state_action_pair).detach() + critic_random.detach()
            d_h = torch.Tensor([[self.timestep - timestep_random]])

            next_state_random_action_pair = torch.cat([next_state_random, self.actor(next_state_random, d_Q, torch.Tensor([[self.timestep-timestep_random]])).detach()], dim=1)
            goal_action_random = goal_action_random.detach()

            Q_next = self.critic(next_state_random_action_pair).detach()
            self.critic.train_step(state_action_pair_random, reward_random + 0.96 * Q_next)


            policy_action, goal_action = self.actor(state_random, d_Q, d_h)
            loss_g = torch.nn.functional.mse_loss(goal_action, goal_action_random)
            loss_a = torch.nn.functional.mse_loss(policy_action, action_random)
            loss_a.backward(retain_graph=True)
            loss_g.backward()
            loss = loss_a + loss_g
            self.actor_optim.step()
            return loss



class HUDQLAgent:
    def __init__(self, action_size=200, mcs=1, goal_size=32):
        self.actor = Brain(action_size=action_size).cuda()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.timestep = 0
        self.maxlen = 20
        self.simulation_num = 3
        self.memory = deque(maxlen=self.maxlen)
        self.reward_memory = deque(maxlen=self.maxlen)

        self.critic = Critic().cuda()
        self.desired_reward = torch.Tensor([[10]]).cuda()
        self.desired_time = torch.Tensor([[10]]).cuda()

    def remember(self, state, action, reward, next_state):
        next_state = next_state.cuda()
        next_action = self.actor(next_state, self.desired_reward, self.desired_time+1)

        critique = self.critic(torch.cat([next_state, next_action], dim=1))
        
        next_state = next_state.cpu()
        next_action = next_action.cpu()
        action = action.cpu()
        

        self.memory.append((state, action, reward, self.timestep, next_state, critique))
        self.reward_memory.append(critique.detach().item())
    def act(self, state):
        desired_time = self.maxlen - self.timestep

        if desired_time == 0:
            self.timestep = 0

        mean, std = np.mean(self.reward_memory), np.std(self.reward_memory)
        desired_reward = np.random.uniform(mean, (mean+std))
        self.desired_reward = torch.Tensor([[desired_reward]]).cuda()
        
        self.desired_time = torch.Tensor([[desired_time]]).cuda()
        
        state = state.cuda()
        
        action = self.actor(state, self.desired_reward, self.desired_time)

        self.timestep += 1

        return action.cpu().detach()

    def train(self, state, action, reward):
        reward = reward
        memory = np.random.choice(len(self.memory), p=softmax(np.array(self.reward_memory)))
        memory = [self.memory[memory]]
        for state_random, action_random, reward_random, timestep_random, next_state_random, critic_random in memory:        
            action_random = action_random.detach()
            state_action_pair_random = torch.cat([state_random.cuda(), action_random.cuda()], dim=1).cuda()
            state_action_pair = torch.cat([state, action], dim=1).cuda()
            
            
            d_Q = self.critic(state_action_pair).detach() + critic_random.detach().cuda()
            d_h = torch.Tensor([[self.timestep - timestep_random]]).cuda()
            next_state_random = next_state_random.cuda()
            
            next_state_random_action_pair = torch.cat([next_state_random, self.actor(next_state_random, d_Q, d_h).detach()], dim=1)

            Q_next = self.critic(next_state_random_action_pair).detach()
            self.critic.train_step(state_action_pair_random, reward_random + 0.96 * Q_next)


            policy_action = self.actor(state_random.cuda().detach(), d_Q, d_h)

            loss_a = torch.nn.functional.mse_loss(policy_action, action_random.cuda())
            Q_loss = -self.critic(torch.cat([state_random.cuda(), policy_action], dim=1))

            loss_a.backward(retain_graph=True)
            Q_loss.backward(retain_graph=True)
            loss = (1.7 * loss_a) + (1.3 * Q_loss)
            self.actor_optim.step()
            return loss
