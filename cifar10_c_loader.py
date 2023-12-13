import os.path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
class Cifarc_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir


        # Get list of image and label paths for each split
        self.train_imgs, self.train_labels = os.path.join(data_dir, "train/")
        # Similar code for test and val data

    def __len__(self):
        return len(self.train_imgs)  # Replace with length of respective split

    def __getitem__(self, idx):
        img_path = self.train_imgs[idx]
        #label = self.train_labels[idx]

        image = Image.open(img_path)

        sample = {"image": image}#, "label": label}
        # Add additional information if needed

        return sample
    #classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

