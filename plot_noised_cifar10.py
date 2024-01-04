import matplotlib.pyplot as plt
import torch
import torch.utils.data
import numpy as np
import pickle

from torchvision.transforms import Compose, ToTensor, Normalize, RandomApply, Lambda, Resize, RandomResizedCrop, InterpolationMode, CenterCrop

def get_stl10_transform(noise_level):
    random_noise = torch.randn(3, 32, 32) * noise_level  # Generate random noise once

    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Lambda(lambda x: x + random_noise)  # Apply the pre-generated random noise
    ])

    return transform

root = '/Data/federated_learning/data/cifar10/cifar-10-batches-py/'
bin_path = root+'data_batch_2'
file_num = 227

def load_cifar10_batch(bin_path):
    with open(bin_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='latin1')

    # Extract features and labels
    features = data_dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(data_dict['labels'])

    return features, labels

# Example usage
features, labels = load_cifar10_batch(bin_path)

noise_levels = np.linspace(0.0, 0.7, 8)

def plot(noise_levels: list, original_img = features[file_num]):
    original_img = original_img
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Effect of Different Noise Levels', fontsize=16)
    for i in range(len(noise_levels)):
        transform = get_stl10_transform(noise_levels[i])
        img_noised = transform(original_img.copy())
        axes[i // 4, i % 4].imshow(img_noised.numpy().transpose(1, 2, 0))
        axes[i // 4, i % 4].set_title('Noise level = ' + str("%.1f" %noise_levels[i]))
        axes[i // 4, i % 4].axis('off')
    plt.savefig('/Data/federated_learning/jetson-intro-to-distillation/data/noise_levels_cifar.png')
    plt.show()



plot(noise_levels,features[file_num])



