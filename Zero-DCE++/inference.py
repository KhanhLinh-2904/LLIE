import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import PIL
import cv2
def is_low_light(image, threshold=50):
    # Load the image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity
    average_intensity = cv2.mean(gray_image)[0]

    # Check if average intensity is below the threshold
    if average_intensity < threshold:
        return True
    else:
        return False

def lowlight(image_path,image_name):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)

	# data_lowlight = data_lowlight.resize((512, 512), PIL.Image.Resampling.LANCZOS)

	data_lowlight = (np.asarray(data_lowlight)/255.0)
	print("data_lowlight: ",data_lowlight.shape)

	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0)
	print("shape of image: ",data_lowlight.shape)
	DCE_net = model.enhance_net_nopool(scale_factor)
	DCE_net.load_state_dict(torch.load('/home/user/low_light_enhancement/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)
	print("shape of enhanced_image: ",enhanced_image.shape)
	end_time = (time.time() - start)

	print(end_time)
	result_path = '/home/user/low_light_enhancement/Zero-DCE++/data/result_printed/'
	result_path = os.path.join(result_path, image_name)
	# print("result_path: ",result_path)
	# print('enhanced_image: ', type(enhanced_image))
	# print('enhanced_image: ', enhanced_image.shape)
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	with torch.no_grad():

		filePath = '/home/user/low_light_enhancement/Zero-DCE++/data/printed/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		threshold = 100

		for file_name in file_list:
			print("file_name:",file_name)
			path_to_image = os.path.join(filePath, file_name)
			# print("path_to_image:",path_to_image)
			img = cv2.imread(path_to_image)
			# if is_low_light(img, threshold):
			sum_time = sum_time + lowlight(path_to_image,file_name)
		print(sum_time)
		

