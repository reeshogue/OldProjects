import pygame
import torch
import cv2
import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

def get_shot(camera):
    ret, frame = camera.read()
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = to_tensor(frame)
    frame = torch.unsqueeze(frame, 0)
    frame = torch.nn.functional.interpolate(frame, size=(200,200))
    return frame

def get_image_for_comparison(file='./assets/*'):
    return torch.load('assets.ptch')

def get_image_for_comparison_v1(file='./assets/*'):
    f = []
    for i in glob.glob(file):
        image = Image.open(i).convert('RGB')
        im =  to_tensor(image)
        im = torch.unsqueeze(im, 0)
        print(im.shape)
        frame = torch.nn.functional.interpolate(im, size=(200,200))
        f.append(frame)
    return f
    
def show_image_pygame(tensor, display, size):
    tensor = torch.squeeze(tensor, 0)
    image = to_pil_image(tensor)
    surf = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    surf = pygame.transform.scale(surf, size)

    display.blit(surf, (0, 0))
    pygame.display.flip()

if __name__ == '__main__':
    y = get_image_for_comparison_v1()
    y_h = get_image_for_comparison()
    y_h.extend(y)
    torch.save(y_h, 'assets.ptch')
