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

def lowlight(image_path,image_name):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0)
	print("shape of image: ",data_lowlight.shape)
	DCE_net = model.enhance_net_nopool(scale_factor)
	DCE_net.load_state_dict(torch.load('/home/user/low_light_enhancement/Zero_DCE_Tiny/snapshots_Zero_DCE_Tiny_new/Epoch14.pth', map_location=torch.device('cpu')))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	result_path = '/home/user/low_light_enhancement/Zero_DCE_Tiny/data/result_Test_Part2/'
	result_path = os.path.join(result_path, image_name)
	print("result_path: ",result_path)
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	with torch.no_grad():

		filePath = '/home/user/low_light_enhancement/Zero-DCE++/data/Test_Part2/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			print("file_name:",file_name)
			path_to_image = os.path.join(filePath, file_name)
			print("path_to_image:",path_to_image)
			sum_time = sum_time + lowlight(path_to_image,file_name)
		print(sum_time)
		

