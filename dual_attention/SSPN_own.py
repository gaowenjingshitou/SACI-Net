import math

import torch.nn as nn
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import functools
from functools import partial
import os, sys
BASE_DIR=r'/home/gwj/Intussption_classification'
BASE_DIR1=r'/home/gwj/Intussption_classification/models'
BASE_DIR2=r'/home/gwj/Intussption_classification/model'
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR1)
sys.path.append(BASE_DIR2)
from model.sync_batchnorm import SynchronizedBatchNorm2d



__all__ = ['SSFPN']


class SSFPN(nn.Module):
	def __init__(self, backbone, pretrained=True, classes=3):
		super(SSFPN, self).__init__()
		
		if backbone.lower() == "resnet18":
			encoder = resnet18(pretrained=pretrained)
			out_channels = 512
		elif backbone.lower() == "resnet34":
			encoder = resnet34(pretrained=pretrained)
			out_channels = 512
		elif backbone.lower() == "resnet50":
			encoder = resnet50(pretrained=pretrained)
			out_channels = 2048
		elif backbone.lower() == "resnet101":
			encoder = resnet101(pretrained=pretrained)
			out_channels = 2048
		elif backbone.lower() == "resnet152":
			encoder = resnet152(pretrained=pretrained)
		else:
			raise NotImplementedError("{} Backbone not implemented".format(backbone))
		
		# self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
		self.conv1_x = encoder.conv1
		self.bn1 = encoder.bn1
		self.relu = encoder.relu
		self.maxpool = encoder.maxpool
		self.conv2_x = encoder.layer1  # 1/4
		self.conv3_x = encoder.layer2  # 1/8
		self.conv4_x = encoder.layer3  # 1/16
		self.conv5_x = encoder.layer4  # 1/32
		self.fab = nn.Sequential(
			conv_block(2048, 256, 3, 1, padding=1, group=256, dilation=1, bn_act=True),
			nn.Dropout(p=0.15))
		
		# self.cfgb = nn.Sequential(
		#     conv_block(512,512,3,2,padding=1,group=512,dilation=1,bn_act=True),
		#     nn.Dropout(p=0.15))
		
		self.cfgb = nn.Sequential(
			conv_block(2048, 512, 3, 2, padding=1, dilation=1, bn_act=True),
			nn.Dropout(p=0.15))
		
		# self.apf1 = PyrmidFusionNet(512, 512, 256, classes=classes)
		# self.apf2 = PyrmidFusionNet(256, 256, 128, classes=classes)
		# self.apf3 = PyrmidFusionNet(128, 128, 64, classes=classes)
		# self.apf4 = PyrmidFusionNet(64, 64, 32, classes=classes)
		
		# self.apf1 = PyrmidFusionNet(2048, 512, 256, 32,classes=classes)
		# self.apf2 = PyrmidFusionNet(256, 256, 128, 16,classes=classes)
		# self.apf3 = PyrmidFusionNet(128, 128, 64,8, classes=classes)
		# self.apf4 = PyrmidFusionNet(64, 64, 32,4, classes=classes)
		#
		self.apf1 = PyrmidFusionNet(2048, 512, 256, 32, classes=classes)
		self.apf2 = PyrmidFusionNet(1024, 256, 128, 16, classes=classes)
		self.apf3 = PyrmidFusionNet(512, 128, 64, 8, classes=classes)
		self.apf4 = PyrmidFusionNet(256, 64, 32, 4, classes=classes)
		
		self.gfu4 = GlobalFeatureUpsample(256,  256,  128)
		self.gfu3 = GlobalFeatureUpsample(128, 128,64)
		self.gfu2 = GlobalFeatureUpsample(64,  64,32)
		self.gfu1 = GlobalFeatureUpsample(32,  32,32)
		
		self.classifier = Classifier(32, classes)
	
	def forward(self, x):
		B, C, H, W = x.size()
		x = self.conv1_x(x)
		x = self.bn1(x)
		x1 = self.relu(x)
		
		x = self.maxpool(x1)
		x2 = self.conv2_x(x)  # torch.Size([1, 256, 56, 56])
		x3 = self.conv3_x(x2)  # torch.Size([1, 512, 28, 28])
		x4 = self.conv4_x(x3)  # torch.Size([1, 1024, 14, 14])
		x5 = self.conv5_x(x4)  # torch.Size([1, 2048, 7, 7])
		
		CFGB = self.cfgb(x5)  # torch.Size([1, 512, 4, 4])
		
		# APF1, cls1 = self.apf1(CFGB, x5)
		# APF2, cls2 = self.apf2(APF1, x4)
		# APF3, cls3 = self.apf3(APF2, x3)
		# APF4, cls4 = self.apf4(APF3, x2)
		
		APF1 = self.apf1(CFGB, x5)#torch.Size([1, 256, 7, 7])
		APF2 = self.apf2(APF1, x4)#torch.Size([1, 128, 14, 14])
		APF3 = self.apf3(APF2, x3)#torch.Size([1, 64, 28, 28])
		APF4 = self.apf4(APF3, x2)#torch.Size([1, 32, 56, 56])
		
		FAB = self.fab(x5)#torch.Size([1, 256, 7, 7])
		
		dec5 = self.gfu4(APF1, FAB)  # torch.Size([1, 128, 7, 7])
		dec4 = self.gfu3(APF2, dec5)#torch.Size([1, 64, 14, 14])
		dec3 = self.gfu2(APF3, dec4)#torch.Size([1, 32, 28, 28])
		dec2 = self.gfu1(APF4, dec3)#torch.Size([1, 32, 56, 56])
		
		classifier = self.classifier(dec2)
		return classifier
		
		# sup1 = F.interpolate(cls1, size=(H, W), mode="bilinear", align_corners=True)
		# sup2 = F.interpolate(cls2, size=(H, W), mode="bilinear", align_corners=True)
		# sup3 = F.interpolate(cls3, size=(H, W), mode="bilinear", align_corners=True)
		# sup4 = F.interpolate(cls4, size=(H, W), mode="bilinear", align_corners=True)
		# predict = F.interpolate(classifier, size=(H, W), mode="bilinear", align_corners=True)
		#
		# if self.training:
		# 	return predict, sup1, sup2, sup3, sup4
		# else:
		# 	return predict


class Attention(nn.Module):
	def __init__(self, size=224, stride=32, heads=8, dim_head=64, dropout=0.):
		super().__init__()
		self.resolution=int(size/stride)
		self.dim=int(math.pow(size/stride,2)) #49
		inner_dim = dim_head * heads#512
		project_out = not (heads == 1 and dim_head == self.dim)
		
		self.heads = heads
		self.scale = dim_head ** -0.5
		
		self.attend = nn.Softmax(dim=-1)
		self.to_qkv = nn.Linear(self.dim, inner_dim * 3, bias=False)
		
		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, self.dim),
			nn.Dropout(dropout)
		) if project_out else nn.Identity()
	
	def forward(self, x1,x2):  #x1= torch.Size([1, 256, 49]),x2=torch.Size([1, 256, 49])
		b_x1, n_x1, _x1, h_x1 = *x1.shape, self.heads
		qkv_x1 = self.to_qkv(x1).chunk(3, dim=-1)
		#q_x1=k_x1=v_x1=torch.Size([1, 8, 256, 64])
		q_x1, k_x1, v_x1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h_x1), qkv_x1)
		
		b_x2, n_x2, _x2, h_x2 = *x2.shape, self.heads
		qkv_x2 = self.to_qkv(x2).chunk(3, dim=-1)
		#q_x2=k_x2=v_x2 =torch.Size([1, 8, 256, 64])
		q_x2, k_x2,v_x2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h_x1), qkv_x2)
		
		
		dots_x1 = einsum('b h i d, b h j d -> b h i j', q_x1, k_x2) * self.scale#torch.Size([1, 8, 256, 256])
		attn_x1 = self.attend(dots_x1)#torch.Size([1, 8, 256, 256])
		out_x1 = einsum('b h i j, b h j d -> b h i d', attn_x1, v_x2)#torch.Size([1, 8, 256, 64])
		out_x1 = rearrange(out_x1, 'b h n d -> b n (h d)')##torch.Size([1, 8, 256, 64])
		out_x1=self.to_out(out_x1)##torch.Size([1, 256, 49)
		out_x1_b,out_x1_c,out_x1_hw=out_x1.size()
		out_x1=out_x1.view(out_x1_b,out_x1_c,self.resolution,self.resolution)#torch.Size([1, 256, 7, 7])
		
		dots_x2 = einsum('b h i d, b h j d -> b h i j', q_x2, k_x1) * self.scale  # torch.Size([1, 8, 256, 256])
		attn_x2 = self.attend(dots_x1)  # torch.Size([1, 8, 256, 256])
		out_x2 = einsum('b h i j, b h j d -> b h i d', attn_x2, v_x1)  # torch.Size([1, 8, 256, 64])
		out_x2 = rearrange(out_x2, 'b h n d -> b n (h d)')  ##torch.Size([1, 8, 256, 64])
		out_x2 = self.to_out(out_x2)  ##torch.Size([1, 256, 49)
		out_x2_b, out_x2_c, out_x1_hw = out_x2.size()
		out_x2 = out_x2.view(out_x2_b, out_x2_c, self.resolution, self.resolution)  # torch.Size([1, 256, 7, 7])
		return out_x1,out_x2
		
		# attn = self.attend(dots)
		#
		# out = einsum('b h i j, b h j d -> b h i d', attn, v)
		# out = rearrange(out, 'b h n d -> b n (h d)')
		


class PyrmidFusionNet(nn.Module):
	def __init__(self, channels_high, channels_low, channel_out,s, classes=11,
	             groups=2):  # channels_high 2048 ；channels_low 512
		super(PyrmidFusionNet, self).__init__()
		self.groups = groups
		
		# self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)
		self.lateral_low = conv_block(channels_high, channels_low, 3, 1, bn_act=True, padding=1)
		
		self.conv_low = conv_block(channels_low, channel_out, 3, 1, bn_act=True, padding=1)
		self.sa = SpatialAttention(channel_out, channel_out)
		
		self.att=Attention(size=224,stride=s,heads=8,dim_head=64, dropout=0.)
		
		
		self.conv_high = conv_block(channels_low, channel_out, 3, 1, bn_act=True, padding=1)
		self.ca = ChannelWise(channel_out)
		
		self.FRB = nn.Sequential(
			conv_block(2 * channels_low, channel_out, 1, 1, bn_act=True, padding=0),
			conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))
		
		self.classifier = nn.Sequential(
			conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
			nn.Dropout(p=0.15),
			conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
		self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)
	
	def forward(self, x_high, x_low):  # x_high=torch.Size([1, 512, 4, 4]) x_low=torch.Size([1, 2048, 7, 7])
		_, _, h, w = x_low.size()  # torch.Size([1, 2048, 7, 7])
		
		lat_low = self.lateral_low(x_low)  # torch.Size([1, 512, 7, 7])
		
		high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear',
		                         align_corners=False)  # torch.Size([1, 512, 7, 7])
		
		concate = torch.cat([lat_low, high_up1], 1)  # torch.Size([1, 1024, 7, 7])
		concate = self.FRB(concate)  # torch.Size([1, 256, 7, 7])
		
		conv_high = self.conv_high(high_up1)  # torch.Size([1, 256, 7, 7])
		conv_low = self.conv_low(lat_low)  # torch.Size([1, 256, 7, 7])
		
		sa = self.sa(concate)## torch.Size([1, 256, 7, 7])
		sa_b, sa_c, sa_h, sa_w = sa.size()
		sa=sa.view(sa_b,sa_c,-1)#torch.Size([1, 256, 49])
		ca = self.ca(concate)#torch.Size([1, 256, 7, 7])
		ca_b, ca_c, ca_h, ca_w = ca.size()
		ca = sa.view(ca_b, ca_c, -1)  # torch.Size([1, 256, 49])
		ca_output,sa_out_put = self.att(ca,sa)  # torch.Size([1, 256, 49])
		
		
		
		mul1 = torch.mul(sa_out_put, conv_high)#torch.Size([1, 256, 7, 7])
		mul2 = torch.mul(ca_output, conv_low)#torch.Size([1, 256, 7, 7])
		
		att_out = mul1 + mul2
		
		# sup = self.classifier(att_out)
		APF = self.apf(att_out)
		#return APF, sup
	
		return APF


class GlobalFeatureUpsample(nn.Module):
	def __init__(self,  in_channels, out_channels, low_channels,red=1):
		super(GlobalFeatureUpsample, self).__init__()
		
		self.conv1 = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
		self.conv2 = nn.Sequential(
			conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
			nn.ReLU(inplace=True))
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv3 = conv_block(out_channels, low_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
	
	def forward(self, x_gui, y_high):#x_gui=torch.Size([1, 256, 7, 7]);y_high==torch.Size([1, 256, 7, 7])
		# h, w = x_gui.size(2), x_gui.size(3)
		x_gui=self.conv1(x_gui)
		#y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
		y_high = self.conv2(y_high)
		y_high = self.avg_pool((y_high))#torch.Size([1, 256, 1, 1])
		
		
		
		out = y_high.expand_as(x_gui) * x_gui#torch.Size([1, 256, 7, 7])
		
		return self.conv3(out)


class conv_block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
	             bias=False):
		super(conv_block, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
		                      padding=padding, dilation=dilation, groups=group, bias=bias)
		self.bn = SynchronizedBatchNorm2d(out_channels)
		self.act = nn.ReLU(inplace=False)
		self.use_bn_act = bn_act
	
	def forward(self, x):
		if self.use_bn_act:
			return self.act(self.bn(self.conv(x)))
		else:
			return self.conv(x)


class Classifier(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Classifier, self).__init__()
		
		self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		self.avgpool = nn.AvgPool2d(56, stride=1)
		self.fc = nn.Linear(in_channels, out_channels)
	def forward(self, x):
		x=self.avgpool(x)#torch.Size([1, 32, 1, 1])
		b,c,h,w=x.shape
		x=x.view(b,-1)#32
		x=self.fc(x)
		
		# self.fc = nn.Linear(in_channels, out_channels)
		
		return x


class SpatialAttention(nn.Module):
	def __init__(self, in_ch, out_ch, droprate=0.15):
		super(SpatialAttention, self).__init__()
		self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
		self.bn_sh1 = nn.BatchNorm2d(in_ch)
		self.bn_sh2 = nn.BatchNorm2d(in_ch)
		self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
		self.drop = droprate
		self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
		self.gamma = nn.Parameter(torch.zeros(1))
	
	def forward(self, x):
		b, c, h, w = x.size()
		
		avgpool_h = F.avg_pool2d(x, [1, w])  # .view(b,c,-1).permute(0,2,1)#torch.Size([1, 256, 1, 7])
		avgpool_h = F.conv2d(avgpool_h, self.conv_sh.weight, padding=0, dilation=1)  # torch.Size([1, 256, 1, 7])
		avgpool_h = self.bn_sh1(avgpool_h)  # torch.Size([1, 256, 7, 1])
		
		avgpool_w = F.avg_pool2d(x, [h, 1])  # torch.Size([1, 256, 1, 7]) # .view(b,c,-1)
		avgpool_w = F.conv2d(avgpool_w, self.conv_sh.weight, padding=0, dilation=1)  # torch.Size([1, 256, 1, 7])
		avgpool_w = self.bn_sh2(avgpool_w)  # torch.Size([1, 256, 1, 7])
		
		att = torch.softmax(torch.matmul(avgpool_h, avgpool_w), 1)#torch.Size([1, 256, 7, 7])
		attt1 = att[:, 0, :, :].unsqueeze(1)#torch.Size([1, 1, 7, 7])
		attt2 = att[:, 1, :, :].unsqueeze(1)##torch.Size([1, 1, 7, 7])
		
		fusion = attt1 * avgpool_h + attt2 * avgpool_w#torch.Size([1, 256, 7, 7])
		out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)#torch.Size([1, 256, 7, 7])
		
		# out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
		out = F.relu(self.gamma * out + (1 - self.gamma) * x)#torch.Size([1, 256, 7, 7])
		return out


class ChannelWise(nn.Module):
	def __init__(self, channel, reduction=4):
		super(ChannelWise, self).__init__()
		self.conv_b = nn.Conv2d(channel*2, channel // 8, 1)
		self.conv_c = nn.Conv2d(channel*2, channel // 8, 1)
		self.conv_d = nn.Conv2d(channel*2, channel, 1)
		self.alpha = nn.Parameter(torch.zeros(1))
		self.softmax = nn.Softmax(dim=-1)
	
	# self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# self.conv_pool = nn.Sequential(
		# 	conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
		# 	conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())
	
	def forward(self, x):
		x_avg_pool2d=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)#torch.Size([1, 256, 7, 7])
		x_max_pool2d=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)#torch.Size([1, 256, 7, 7])
		x_cat=torch.cat([x_avg_pool2d,x_max_pool2d],axis=1)##torch.Size([1, 512, 7, 7])
		
		batch_size, _, height, width = x_cat.size()
		feat_b = self.conv_b(x_cat).view(batch_size, -1, height * width).permute(0, 2, 1)#torch.Size([1, 49, 32])
		feat_c = self.conv_c(x_cat).view(batch_size, -1, height * width)#torch.Size([1, 32, 49])
		attention_s = self.softmax(torch.bmm(feat_b, feat_c))#torch.Size([1, 49, 49])
		feat_d = self.conv_d(x_cat).view(batch_size, -1, height * width)#torch.Size([1, 256, 49])
		feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)#torch.Size([1, 256, 7, 7])
		out = self.alpha * feat_e + (1-self.alpha)*x#torch.Size([1, 256, 7, 7])
		# y = self.avg_pool(x)
		# y = self.conv_pool(y)
		return out
		#return x * y


if __name__ == "__main__":
	# input1 = torch.rand(2,3,360,480)
	# model = SSFPN("resnet18")
	# summary(model, torch.rand((2,3,360,480)))
	input1 = torch.rand(1, 3, 224, 224)
	print(input1)
	model = SSFPN("resnet152")
	# summary(model, torch.rand((2, 3, 360, 480)))
	output = model(input1)
	print(output.shape)

# python train_cityscapes.py --dataset camvid --model MSFFNet --batch_size 4 --max_epochs 300 --train_type trainval --lr 1e-3



