import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

def is_cuda_available():
    # CUDA?
    cuda = torch.cuda.is_available()
    return cuda

def plot_image(images):
    
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

def plot_images(num_of_images, images):
    figure = plt.figure()
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

     