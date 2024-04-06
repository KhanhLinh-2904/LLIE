# import os
# import shutil

# # Specify the input and output folders
# input_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/Dataset_Part2'
# output_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/Test_Part2/'


# def list_folders(directory):
#     folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
#     return folders

# directory_path = "/home/user/low_light_enhancement/Zero-DCE++/data/Dataset_Part2"
# folders_in_directory = list_folders(directory_path)
# # print(len(folders_in_directory))
# cnt_files_3 = 0
# cnt_files_4 = 0
# cnt_img = 0
# for folder in folders_in_directory:
#     # Loop through all files in the input folder
#     folder_path = os.path.join(directory_path, folder)
#     folder_path = folder_path + '/'
#     # print(folder_path)
#     if folder == 'Label':
#         continue
#     count = len(os.listdir(folder_path))
#     cnt_img += count
# print("cnt_img: ",cnt_img)
# #     if count < 9:
# #         cnt_files_3 += 1
# #     else:
# #         cnt_files_4 += 1

# #     iter = 0
# #     if count < 9:
# #         for filename in os.listdir(folder_path):
# #             with open(os.path.join(folder_path, filename), 'rb') as f:
# #                 image_data = f.read()
# #             filename = output_folder + folder + '_' + filename.split('.')[0] + '.jpg'
# #             with open(os.path.join(output_folder, filename), 'wb') as f:
# #                 f.write(image_data)
# #             iter += 1
# #             if iter >= 3:
# #                 break
# #     else:
# #         for filename in os.listdir(folder_path):
# #             with open(os.path.join(folder_path, filename), 'rb') as f:
# #                 image_data = f.read()
# #             filename = output_folder + folder + '_' + filename.split('.')[0] + '.jpg'
# #             with open(os.path.join(output_folder, filename), 'wb') as f:
# #                 f.write(image_data)
# #             iter += 1
# #             if iter >= 4:
# #                 break
# # print("cnt_files_3: ",cnt_files_3)
# # print("cnt_files_4: ",cnt_files_4)
# # #####################################3
# # count number of folder file


# # print("Images have been successfully copied to the output folder.")


# ##################################################
# # output_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/Test_Part2/'

# # print(len(os.listdir(output_folder)))
# #Read all images and partition into 2 folders: training dataset and validation dataset

# # import torch
# # from torch.utils.data import Dataset, DataLoader

# # import glob
# # dataset = glob.glob('/home/user/low_light_enhancement/Zero-DCE++/data/ori_data/*.jpg')
# # print(len(dataset))

# # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# output_folder = '/home/user/low_light_enhancement/Zero-DCE++/data/Dataset_Part2/Label'
# out = '/home/user/low_light_enhancement/Zero-DCE++/data/label_Test_Part2'
# count = 0
# for filename in os.listdir(output_folder):
#     # Read the image file
#     count += 1
#     with open(os.path.join(output_folder, filename), 'rb') as f:
#         image_data = f.read()

#     # Save the image file to the output folder
#     filename = filename.split('.')[0] + '.jpg'
#     # print(filename)
#     with open(os.path.join(out, filename), 'wb') as f:
#         f.write(image_data)
#     # if filename.endswith('.JPG') or filename.endswith('.PNG'):  # Add more image formats if needed     

# print(count)

import numpy as np
import torch

data = torch.Tensor([[1,2,3,4],[1,2,3,4]])
arr = data.numpy()
print(torch.mean(data))
print(np.mean(arr))