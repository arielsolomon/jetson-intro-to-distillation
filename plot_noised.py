import matplotlib.pyplot as plt
import torch
import torch.utils.data
import numpy as np
from PIL import Image

from torchvision.transforms import Compose, ToTensor, Normalize, RandomApply, Lambda, Resize, RandomResizedCrop, InterpolationMode, CenterCrop

def get_stl10_transform(noise_level):
    random_noise = torch.randn(3, 96, 96) * noise_level  # Generate random noise once

    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Lambda(lambda x: x + random_noise)  # Apply the pre-generated random noise
    ])

    return transform

root = '/Data/federated_learning/jetson-intro-to-distillation/data/stl10/stl10_binary/'
bin_path = root+'train_X.bin'
file_num = 227
def load_bin(bin_path, file_num):

    img3 = np.fromfile(bin_path, dtype=np.uint8)
    img2 = np.reshape(img3,(int(img3.shape[0]/(3*96*96)),3,96,96))
    img1 = np.transpose(img2[file_num,:,:,:], (2, 1, 0))
    return img1

noise_levels = np.linspace(0.0, 0.7, 8)

def plot(noise_levels: list, original_img = load_bin(bin_path, file_num)):
    original_img = original_img
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Effect of Different Noise Levels', fontsize=16)
    for i in range(len(noise_levels)):
        transform = get_stl10_transform(noise_levels[i])
        img_noised = transform(original_img.copy())
        axes[i // 4, i % 4].imshow(img_noised.numpy().transpose(1, 2, 0))
        axes[i // 4, i % 4].set_title('Noise level = ' + str("%.1f" %noise_levels[i]))
        axes[i // 4, i % 4].axis('off')
    plt.savefig('/Data/federated_learning/jetson-intro-to-distillation/data/noise_levels_stl10.png')
    plt.show()



plot(noise_levels)



