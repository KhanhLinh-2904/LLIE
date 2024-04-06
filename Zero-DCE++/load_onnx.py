import onnxruntime
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F 
import model
import os
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import argparse
import time
import dataloader
import model
import numpy as np
import torchvision 
from PIL import Image
import glob
import time
import PIL
from matplotlib import pyplot as plt
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def lowlight(image_path,image_name):
    scale_factor = 12
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)
	
    # print(data_lowlight.shape)
    torch_model = model.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(torch.load('/home/user/low_light_enhancement/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
	
    onnx_program = torch.onnx.dynamo_export(torch_model, data_lowlight)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(data_lowlight)
    
    onnx_program.save("ZeroDCE++1.onnx")

    ort_session = onnxruntime.InferenceSession("ZeroDCE++1.onnx", providers=['CPUExecutionProvider'])
    
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    start = time.time()
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    end_time = (time.time() - start)
    print(type(onnxruntime_outputs))
    print('onnxruntime_outputs[0]: ', onnxruntime_outputs[0].shape)
	
    output1 = onnxruntime_outputs[0].reshape(onnxruntime_outputs[0].shape[1], onnxruntime_outputs[0].shape[2], onnxruntime_outputs[0].shape[3])
    output2 = onnxruntime_outputs[1].reshape(onnxruntime_outputs[1].shape[1], onnxruntime_outputs[1].shape[2], onnxruntime_outputs[1].shape[3])

    # f, axarr = plt.subplots(1,2, figsize=(10, 5))
    # show output 1
    red_channel = output1[0]
    green_channel = output1[1]
    blue_channel = output1[2]
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    # axarr[0].imshow(rgb_image)
    result_path = '/home/user/low_light_enhancement/Zero-DCE++/data/result_fas_dark_face/'
	
    result_path = os.path.join(result_path, image_name)
    # plt.imshow(rgb_image)
    # print(rgb_image.shape)
    # print(type(rgb_image))
    plt.imsave(result_path, rgb_image)
    # show output 2
    # red_channel = output2[0]
    # green_channel = output2[1]
    # blue_channel = output2[2]
    # rgb_image2= np.stack([red_channel, green_channel, blue_channel], axis=-1)
    # axarr[1].imshow(rgb_image2)
    # plt.axis('off')  # Hide axes
    # plt.show()
    # enhanced_image = torch.from_numpy(onnxruntime_outputs[0])
    # result_path = '/home/user/low_light_enhancement/Zero-DCE++/data/result_Test_Part2_onnx/'
	
    # result_path = os.path.join(result_path, image_name)
	
    # print("result_path: ",result_path)
	
    # torchvision.utils.save_image(enhanced_image, result_path)
    
    return end_time
# torch_input = torch.randn(1, 3, 1200, 900)

# onnx_model = onnx.load("Zero_DCE++.onnx")
# torch_model = model.enhance_net_nopool(1)
# torch_model.load_state_dict(torch.load('/home/user/low_light_enhancement/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
# onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
# onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
# print(f"Input length: {len(onnx_input)}")
# print(f"Sample input: {onnx_input}")

# # print(type(torch_model))


# torch_input = torch.randn(1, 3, 3648, 5472)


# ort_session = onnxruntime.InferenceSession("./Zero_DCE++.onnx", providers=['CPUExecutionProvider'])

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

# onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
# print("onnxruntime_outputs: ", type(onnxruntime_outputs))
# print("onnxruntime_outputs: ", type(onnxruntime_outputs[0]))
# print("onnxruntime_outputs: ", onnxruntime_outputs[0].shape)
# print("onnxruntime_outputs: ", onnxruntime_outputs[1].shape)
# from matplotlib import pyplot as plt
# output1 = onnxruntime_outputs[0].reshape(3, 1200, 900)
# print("output1: ", output1.shape)

# red_channel = output1[0]
# green_channel = output1[1]
# blue_channel = output1[2]
# rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
# plt.imshow(rgb_image)
# plt.axis('off')  # Hide axes
# plt.show()


# output2 = onnxruntime_outputs[1].reshape(3, 1200, 900)
# print("output1: ", output2.shape)
# red_channel = output2[0]
# green_channel = output2[1]
# blue_channel = output2[2]
# rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
# plt.imshow(rgb_image)
# plt.axis('off')  # Hide axes
# plt.show()

# output3 = output2 - output1
# red_channel = output3[0]
# green_channel = output3[1]
# blue_channel = output3[2]
# rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
# plt.imshow(rgb_image)
# plt.axis('off')  # Hide axes
# plt.show()


# torch_outputs = torch_model(torch_input)
# torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

# assert len(torch_outputs) == len(onnxruntime_outputs)
# for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
#     torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

# print("PyTorch and ONNX Runtime output matched!")
# print(f"Output length: {len(onnxruntime_outputs)}")
# print(f"Sample output: {onnxruntime_outputs}")
# print("onnxruntime_outputs: ", len(onnxruntime_outputs))


if __name__ == '__main__':

	with torch.no_grad():

		filePath = '/home/user/low_light_enhancement/Zero-DCE++/data/fas_dark_face/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			print("file_name:",file_name)
			path_to_image = os.path.join(filePath, file_name)
			print("path_to_image:",path_to_image)
			sum_time = sum_time + lowlight(path_to_image,file_name)
		print(sum_time)
		