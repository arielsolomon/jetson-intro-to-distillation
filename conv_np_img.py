import numpy as np
from PIL import Image
import random
import glob
import os
import shutil

root = '/home/user1/ariel/federated_learning/data/cifar10_c/'
dest = '/home/user1/ariel/federated_learning/data/cifar10_c/gaussian_noise/'
np_file = np.load(root+'gaussian_noise.npy')
conv = False
if conv:
    for i in range(np_file.shape[0]):
        img = Image.fromarray(np_file[i,:,:,:])
        img.save(dest+str(i)+'.jpg')
file_list = glob.glob(os.path.join(dest, '*.jpg'))
shufled_list = random.sample(file_list, len(file_list))
val_rath, test_rath, train_rath = 0.15, 0.15, 0.7
train_l, val_l, test_l = shufled_list[:int(train_rath*len(shufled_list))]\
    ,shufled_list[int(train_rath*len(shufled_list)):int(train_rath*len(shufled_list)+test_rath*len(shufled_list))]\
    ,shufled_list[int(train_rath*len(shufled_list)+test_rath*len(shufled_list)):len(shufled_list)]
fold_list = []
[fold_list.append(dest+fold) for fold in ['train/','test/','val/']]
for fold in fold_list:
    if not os.path.exists(fold):
        os.mkdir(fold)
file_list = ['train_l', 'val_l', 'test_l']
for name in file_list:
    value = globals()[name]
    for file in value:
        shutil.copy2(file, dest+name[:-2]+'/')

