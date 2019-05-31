import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def denorm(x):
    """ Denormalize inputs from in N(0, 1) to U(0, 1)
    """
    x = (x + 1)/2
    return x.clamp(0, 1)

def sample_generator(g_model, noise, transform=None, no_samples=20, device='cpu'):
    """ Sample images from a Generator and display them in a grid
    """
    fake_images = g_model(noise).to(device)
    fake_images = denorm(fake_images)
    
    if transform != None:
        fake_images = transform(fake_images)
    
    print(vutils.make_grid(fake_images.detach(), padding=2, normalize=True).size())
    plt.imshow(np.transpose(vutils.make_grid(fake_images.detach(), padding=2, normalize=True), (1, 2, 0)))
    plt.pause(0.5)
    plt.close()

def sample_z(size, device='cpu'):
    """ Sample Z as gaussian noise 
    """
    return torch.randn(size, device=device)

def instance_noise():
    