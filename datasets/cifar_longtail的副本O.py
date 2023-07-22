# -*- coding: utf-8 -*-
"""
# @file name  : flower_102.py
# @author     : https://github.com/TingsongYu
# @date       : 2021年4月22日
# @brief      : flower 102数据集读取
"""
import cv2
from config.cifar_config import cfg
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
#-*- coding: UTF-8 -*-
from utils.fmix import sample_mask, make_low_freq_image, binarise_mask
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
from torch import nn
import os
import time
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from numpy import asarray

import cv2
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO

import  pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_data = pd.read_csv("/home/gwj/Intussption_classification/data_tools/train_data_2022_0421.csv", index_col=0)  # 3030 rows × 3 columns
val_data = pd.read_csv("/home/gwj/Intussption_classification/data_tools/val_data_2022_0421.csv", index_col=0)  # 312 rows × 3 columns
test_data = pd.read_csv("/home/gwj/Intussption_classification/data_tools/test_data_2022_0421.csv", index_col=0)  # 381 rows × 3 columns
#test_data = pd.read_csv(os.path.join('/home/gwj/Intussption_classification/人机对比图片', 'images.csv'),
                             # names=['Inhosp_No', 'img', 'label'],
                             #converters={'Inhosp_No':str})
#images_csv1=test_data.set_index("Inhosp_No")
#print("images_csv1",images_csv1)

CFG = {
  
    'img_size': 224,
    'verbose_step': 1,
}

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



class CifarDataset(Dataset):
    names = ('normal','sleeve_sign','concentric_circle_sign')
   # names=('1concentric_circle_sign','2sleeve_sign','3Surgical indications','4normal')
    cls_num = len(names)

    def __init__(self,root_dir,mode,do_fmix=True,fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True,
                     'reformulate': False
                 },
                 do_cutmix=True,
                 cutmix_params={
                     'alpha': 1,
                 },transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []      # 定义list用于存储样本路径、标签
        self._get_img_info(mode)
        self.nums_per_cls = self._get_img_num_per_cls()
        self.error_imgs=[]
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        



        self.name2label = {}  # "sq...":0
        for name in CifarDataset.names:
            #print(name)

            if not os.path.isdir(os.path.join(root_dir, name)):
                print("不是目录", os.path.join(root_dir, name))
                continue
            elif os.path.exists(os.path.join(root_dir, '.ipynb_checkpoints')):
                os.removedirs(os.path.join(root_dir, '.ipynb_checkpoints'))
            elif os.path.exists(os.path.join(root_dir, '.ipynb_checkpoints')):
                os.removedirs(os.path.join(root_dir, '.ipynb_checkpoints'))
            self.name2label[name] = len(self.name2label.keys())
        print("name2label", self.name2label)

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
       # print("path_img",path_img)
       # with open(path_img,'rb') as f:
           # f=f.read()
       # f=f+B'\xff'+B'\xd9'
       # img = Image.open(BytesIO(f)).convert('RGB')
        #ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(path_img).convert('RGB')
            img = asarray(img)
            
            img = img.type(torch.FloatTensor)
          

        except:
            pass
        if self.transform is not None:
            img = self.transform(image=img)['image']
            #img = self.transform(image=img)

        if self.do_fmix:#and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():

                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']), 0.6, 0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

                #fmix_ix = np.random.choice(self.df.index, size=1)[0]
                #fmix_img = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))
                fmix_ix = np.random.choice(train_data.index, size=1)[0]
                #print("fmix_ix",fmix_ix)
                fmix_img = train_data.iloc[fmix_ix]["img"]
                
               # print("fmix_img",fmix_img)
                

                if self.transform:
                    fmix_img = Image.open(fmix_img).convert('RGB')
                    fmix_img = asarray(fmix_img)
                    fmix_img = self.transform(image=fmix_img)['image']
                    #fmix_img = self.transform(image=fmix_img)

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img
                img = img.type(torch.FloatTensor)

                # print(mask.shape)

                # assert self.output_label==True and self.one_hot_label==True

                # mix target
                #fmix_img = train_data.iloc[fmix_ix]["img"]
                rate = mask.sum() / CFG['img_size'] / CFG['img_size']
              #  target = rate * label + (1. - rate) * self.labels[fmix_ix]
                label = rate * label + (1. - rate) * train_data.iloc[fmix_ix]["label"]# <class 'numpy.float64'>
              #  print('type',type(label))
                
                #label = torch.Tensor(label)

                #label = label.long()
                #label = torch.LongTensor(label)
            
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            # print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(train_data.index, size=1)[0]
                cmix_img=train_data.iloc[cmix_ix]["img"]
               # cmix_img = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                
                if self.transform:
                    cmix_img = Image.open(cmix_img).convert('RGB')
                    cmix_img = asarray(cmix_img)
                    cmix_img = self.transform(image=cmix_img)['image']
                    #cmix_img = self.transform(image=cmix_img)

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                #target = rate * target + (1. - rate) * self.labels[cmix_ix]
                label = rate * label + (1. - rate) * train_data.iloc[cmix_ix]["label"]
               # label =  torch.Tensor(label)
               
               # label = label.long()
                img = img.type(torch.FloatTensor)
                
              # target = rate * label + (1. - rate) * train_data.iloc[fmix_ix]["label"]
           
        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        #self.error_imgs.to_csv("error_imgs.csv")
        return len(self.img_info)

    def _get_img_info(self,mode):
        if mode=="train":#7116
            for i in range(train_data.shape[0]):
                path_img = train_data.iloc[i]["img"]
                label = train_data.iloc[i]["label"]

                self.img_info.append((path_img, int(label)))
            random.shuffle(self.img_info)  # 将数据顺序打乱
        elif mode == "val":
            for i in range(val_data.shape[0]):
                path_img = val_data.iloc[i]["img"]
                label = val_data.iloc[i]["label"]

                self.img_info.append((path_img, int(label)))
            random.shuffle(self.img_info)  # 将数据顺序打乱
        else:
            for i in range(test_data.shape[0]):
                path_img = test_data.iloc[i]["img"]
                label = test_data.iloc[i]["label"]

                self.img_info.append((path_img, int(label)))
            random.shuffle(self.img_info)  # 将数据顺序打乱

    def _get_img_num_per_cls(self):
        """
        依长尾分布计算每个类别应有多少张样本
        :return:
        """
        img_num_per_cls = []
        for item in CifarDataset.names:
            num = int(len(os.listdir(os.path.join(self.root_dir, item))))
            img_num_per_cls.append(int(num))
        return img_num_per_cls
if __name__ == "__main__":

    #root_dir = r"../../MTX_20210809/"
    root_dir = r"../data/"
   # train_dataset = CifarDataset(root_dir,mode="test")
    train_dataset=CifarDataset(root_dir=root_dir, transform=cfg.transforms_train, mode="train",do_fmix=True, do_cutmix=True)
    
   # print(len(train_dataset))
   # print(next(iter(train_dataset)))

    #print(train_dataset.nums_per_cls) #[3861, 5633]

