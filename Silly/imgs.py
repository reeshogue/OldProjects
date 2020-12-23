import pygame
import torch
import cv2
import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

def get_shot(camera):
    for file in glob.glob('./assets/*'):
        
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = to_tensor(frame)
        frame = torch.unsqueeze(frame, 0)
        frame = torch.nn.functional.interpolate(frame, size=(200,200))
    return frame
