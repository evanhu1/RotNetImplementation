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
    def __init__(self, data_dir, validation=False):
        #gets the data from the directory
        print(data_dir)
        self.file_list = glob.glob(data_dir + ('test*' if validation else 'data*'))
        print(self.file_list)
        self.dict_list = []
        self.dict = defaultdict(list)
        for file in self.file_list:
            with open(file, 'rb') as pickle_file:
                self.dict_list.append(pickle.load(pickle_file, encoding='bytes'))
        for d in self.dict_list:
            for key, value in d.items():
                self.dict[key].extend(value)
        self.image_list = np.array(self.dict[b'data']).reshape(10000 if validation else 50000, 3, 32, 32).transpose(0, 2, 3, 1)
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
        return (image_tensor, np.stack((np.array([1,0,0,0]),
                                np.array([0,1,0,0]),
                                np.array([0,0,1,0]),
                                np.array([0,0,0,1]))))

    def __len__(self):
        return self.data_len
