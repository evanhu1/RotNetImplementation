import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
from torchvision import transforms, utils

'''
Pytorch uses datasets and has a very handy way of creating dataloaders in your main.py
Make sure you read enough documentation.
'''

class Data(Dataset):
    def __init__(self, data_dir):
        #gets the data from the directory
        self.file_list = glob.glob(data_dir+'data*')
        self.dict_list = []
        self.dict = defaultdict(list)
        for file in self.file_list:
            with open(file, 'rb') as pickle_file:
                self.dict_list.append(pickle.load(pickle_file, encoding='bytes'))
        for d in self.dict_list:
            for key, value in d.items():
                self.dict[key].extend(value)
        self.image_list = np.array(self.dict[b'data']).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.dict[b'labels'])
        #calculates the length of image_list
        self.data_len = len(self.image_list)
        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        image = Image.fromarray(single_image_path)
        images = np.array([np.array(image.rotate(n)) for n in [0, 90, 180, 270]])
        # Do some operations on image
        # Convert to numpy, dim = 28x28
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        '''
		#TODO: Convert your numpy to a tensor and get the labels
		'''
        image_tensor = torch.from_numpy(images).float()
        return (image_tensor, self.labels[index])

    def __len__(self):
        return self.data_len

d = Data("/Users/evanhu/code/SP21-NMEP/hw6/data/cifar-10-batches-py/")
c = 0
for p in d.__getitem__(0)[0]:
    print(p.numpy().transpose(2, 0, 1).shape)
    Image.fromarray(p.numpy()).save("image" + c)
    c += 1