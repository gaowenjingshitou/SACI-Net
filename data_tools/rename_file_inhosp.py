
import torch
import shutil
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms


files=os.listdir("../data/normal/")
for item in files:
    src_path=os.path.join("../data/normal/",item)
    #print('src_path',src_path)
    item_0=item.split("-")[0]
    #print('item_0',item_0)
    item_1=item.split("-")[1]

    #print('item_1',item_1)
    item_2=item.split("-")[2]
    # print('item_2',item_2)
    #item_1_0=item_1.split("_")[0]
    #print('item_1_0',item_1_0)
    #item_1_1=item_1.split("_")[1]
    # print('item_1_1',item_1_1)
    dst_path=os.path.join("../data/normal/",item_0+item_1+"-"+item_2)

   # print('dst_path',dst_path)
    os.rename(src_path,dst_path)
    print("success")


   #
   #
   #  item_2=item.split("-")[2]

   # dst_path=os.path.join("../data/normal/",item_1+"-"+item_0+"-"+item_2)
    #print('dst_path',dst_path)


    #print(src_path)
    # item_=os.path.splitext(item)[0]
    # dst_path=os.path.join("../data/sleeve_sign/",item_+".jpg")
    # #print(dst_path)
    # os.rename(src_path,dst_path)
    # print("success")