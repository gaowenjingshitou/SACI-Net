
import torch
import shutil
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms

files=os.listdir("../data/sleeve_sign/")
for item in files:
    src_path=os.path.join("../data/sleeve_sign/",item)
    #print(src_path)
    item_=os.path.splitext(item)[0]
    dst_path=os.path.join("../data/sleeve_sign/",item_+".jpg")
    #print(dst_path)
    os.rename(src_path,dst_path)
    print("success")