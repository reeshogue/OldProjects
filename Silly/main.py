import pygame
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix

from utils import get_shot, show_image_pygame, get_image_for_comparison
from agent import HUDQLAgent as Agent
from brain_v2 import Discriminator as Critic

imgs = get_image_for_comparison()
pygame.init()
disp = pygame.display.set_mode((500, 500), pygame.RESIZABLE)
size = (500,500)
clock  = pygame.time.Clock()
camera = cv2.VideoCapture(0)

next_frame = get_shot(camera)

agent = Agent(goal_size=len(imgs))
critic = Critic()

prev_action = torch.ones((1,3,200,200))
reward = 0.0

agent.remember(next_frame, prev_action, reward, next_frame)

reward_history = []
Q_loss_history = []
acc = torch.Tensor([0])
pacc = 1e-4
sacc = 1e-3

facc = 1.0

while True:
    print(torch.cuda.memory_allocated() / (10 ** 9), "GB, GPU")
    state = next_frame
    action = agent.act(state)

    show_image_pygame(action, disp, size)
    
    reward = critic(state).item()
    next_frame = get_shot(camera)
    
    if np.random.choice([True, False]) and np.random.choice([True, False]):

        loss = agent.train(state, action, reward)
        Q_loss_history.append(loss)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                print("reward_history")
                plt.plot(np.linspace(0, int(acc.item()), num=int(acc.item())), reward_history)
                plt.show()
                print("Q_loss_history")
                plt.plot(np.linspace(0, int(acc.item()), num=int(acc.item())), Q_loss_history, 'g')
                plt.show()
                quit()
        if event.type == pygame.VIDEORESIZE:
            size = event.size
            pygame.display.set_mode(size, pygame.RESIZABLE)
            
                
    critic.train_step(state, torch.zeros(1,1))
    critic.train_step((imgs[np.random.choice(len(imgs))]).detach(), torch.ones(1,1))

    agent.remember(state, action, reward, next_frame)
    prev_action = action
    reward_history.append(reward)
    acc += 1