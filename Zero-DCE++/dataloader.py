import os
import sys
import PIL

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
from torch.utils.data import random_split


generator = torch.Generator()
generator.manual_seed(1143)
random.seed(1143)

import warnings
warnings.filterwarnings('ignore')

def populate_train_list(lowlight_images_path):


	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	random.shuffle(image_list_lowlight)
	# print(image_list_lowlight)
	# Calculate the sizes of training and testing datasets
	test_size = 600
	train_size = len(image_list_lowlight) - test_size
	# Split the dataset into training and testing subsets
	train_dataset, test_dataset = random_split(image_list_lowlight, [train_size, test_size],generator=generator)

	
	
	return train_dataset,test_dataset

	

class lowlight_loader(data.Dataset):

	def __init__(self, dataset):

		self.train_list = dataset
		self.size = 512

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


	def __getitem__(self, index):
		# print(index)
		data_lowlight_path = self.data_list[index]
		# if type(data_lowlight_path) == type("abc"):
			# print(1)
		label = data_lowlight_path[0:]
		label = label[::-1]
		label = label.split('/')[0]
		label = label[::-1]
		label = label.split('_')[0]
		# print(label)

		label_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/label_ori_data/'

		label =	label_folder + label + '.jpg'
		label_lowlight = Image.open(label)
		label_lowlight = label_lowlight.resize((self.size,self.size), PIL.Image.Resampling.LANCZOS)
		label_lowlight = (np.asarray(label_lowlight)/255.0) 
		label_lowlight = torch.from_numpy(label_lowlight).float()
		# label.show()
		# label_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size,self.size), PIL.Image.Resampling.LANCZOS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return (data_lowlight.permute(2,0,1), label_lowlight.permute(2,0,1))

	def __len__(self):
		return len(self.data_list)

