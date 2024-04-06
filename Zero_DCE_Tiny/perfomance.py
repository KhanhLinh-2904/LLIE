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


def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	print("h and w: ",h,w)
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0)

	
	start = time.time()
	DCE_net(data_lowlight)
	end_time = (time.time() - start)

	print(end_time)
	return end_time

if __name__ == '__main__':
	scale_factor = 12
	DCE_net = model.enhance_net_nopool(scale_factor)
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
	with torch.no_grad():

		filePath = '/home/linhhima/low_light_enhancement/Zero-DCE++/data/test_data/real/11_0_.png'
		sum_time = 0
		for i in range(0, 110):
			inference_time = lowlight(filePath)
			if i < 10:
				continue
			sum_time = sum_time + inference_time
		print("Perfomance Zero_DCE++: ", sum_time/100)
		

