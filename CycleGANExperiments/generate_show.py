import torchvision as Tv
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_show(generator, data, show_data_plain=True):
    if show_data_plain:
        generation = data
    else:
        generation = generator((data.cpu()))[0].cpu().detach()
        
    plt.imshow(np.transpose(Tv.utils.make_grid(generation.detach(), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
