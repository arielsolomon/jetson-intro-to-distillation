import matplotlib.pyplot as plt
import torch
import torch.utils.data
import numpy as np
from PIL import Image

from torchvision.transforms import Compose, ToTensor, Normalize, RandomApply, Lambda, Resize, RandomResizedCrop, InterpolationMode, CenterCrop


root = '/Data/federated_learning/jetson-intro-to-distillation/data/stl10/stl10_binary/'
bin_path = root+'train_X.bin'
file_num = 252
noise_levels = np.linspace(0.0, 0.7, 8)
def get_stl10_transform(noise_level):
    random_noise = torch.randn(3, 96, 96) * noise_level  # Generate random noise once

    transform = Compose([
        ToTensor(),
        Lambda(lambda x: x + random_noise)  # Apply the pre-generated random noise
        ,Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform
def gaussian_noise_robustness(x, severity=0):
    c = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7][severity]#[.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1),c# * 255, c
def load_bin(bin_path, file_num):

    img3 = np.fromfile(bin_path, dtype=np.uint8)
    img2 = np.reshape(img3,(int(img3.shape[0]/(3*96*96)),3,96,96))
    img1 = np.transpose(img2[file_num,:,:,:], (2, 1, 0))
    return img1

def plot_gaus_torchvision(noise_levels: list, original_img = load_bin(bin_path, file_num)):
    original_img = original_img
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(plot_gaus_torchvision.__name__+' noise levels' , fontsize=16)
    for i in range(len(noise_levels)):
        transform = get_stl10_transform(noise_levels[i])
        img_noised = transform(original_img.copy())
        axes[i // 4, i % 4].imshow(img_noised.numpy().transpose(1, 2, 0))
        axes[i // 4, i % 4].set_title('Noise level = ' + str("%.1f" %noise_levels[i]))
        axes[i // 4, i % 4].axis('off')
    plt.savefig('/Data/federated_learning/jetson-intro-to-distillation/data/noise_added_before_Norm_stl10.png')
    plt.show()

def plot_gaus_rubustness(original_img = load_bin(bin_path, file_num)):
    c_ind = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    original_img = original_img
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(plot_gaus_rubustness.__name__+' noise levels', fontsize=16)
    for i in range(len(c_ind)):
        img_noised, c = gaussian_noise_robustness(original_img.copy(),i)
        axes[i // 4, i % 4].imshow(img_noised)#/255)#.transpose(1, 2, 0))
        axes[i // 4, i % 4].set_title(str(c))
        axes[i // 4, i % 4].axis('off')
    plt.savefig('/Data/federated_learning/jetson-intro-to-distillation/data/rubstess_gaussian_noise_before_norm_stl10.png')
    plt.show()

plot_gaus_torchvision(noise_levels)
#plot_gaus_rubustness(load_bin(bin_path, file_num))



