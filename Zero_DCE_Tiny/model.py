import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, silu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if silu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True) if silu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor):
		super(enhance_net_nopool, self).__init__()
		self.silu = nn.SiLU(inplace=True)
		self.scale_factor = scale_factor
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 16

#   zerodce DWC + p-shared
		self.e_conv1 = GhostModule(3,number_f*2) #Conv 1 : input=3, out=32
		self.e_conv2 = GhostModule(number_f*2,number_f) #Conv 2 : input=32, out=32
		self.e_conv3 = GhostModule(number_f,number_f) #Conv 3 : input=32, out=32
		self.e_conv4 = GhostModule(number_f,number_f*2) #Conv 4 : input=32, out=32
		self.e_conv5 = GhostModule(number_f*3,number_f*2)  #Conv 5 : input=32*2, out=32
		self.e_conv6 = GhostModule(number_f*3,number_f*2) #Conv 6 : input=32*2, out=32
		self.e_conv7 = GhostModule(number_f*4,number_f*3) #Conv 7 : input=32*2, out=32

		self.e_conv8 = CSDN_Tem(number_f*5,3) #Conv 8: input=32*2, out=3

	def enhance(self, x,x_r):

		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
		x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + x_r*(torch.pow(x,2)-x)	
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image = x + x_r*(torch.pow(x,2)-x)	

		return enhance_image
		
	def forward(self, x):
		if self.scale_factor==1:
			x_down = x
		else:
			x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

		x1 = self.e_conv1(x_down)
		x2 = self.e_conv2(x1)
		x3 = self.e_conv3(x2)
		x4 = self.e_conv4(x3)
		x5 = self.e_conv5(torch.cat([x3,x4],1))
		x6 = self.e_conv6(torch.cat([x5,x2],1))
		x7 = self.e_conv7(torch.cat([x6,x1],1))
		x_r = F.tanh(self.e_conv8(torch.cat([x7,x1],1)))
		if self.scale_factor==1:
			x_r = x_r
		else:
			x_r = self.upsample(x_r)
		enhance_image = self.enhance(x,x_r)
		return enhance_image,x_r
