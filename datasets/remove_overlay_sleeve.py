
import os
import random
import shutil

from PIL import Image
from torch.utils.data import Dataset


sleeve=os.listdir("/home/gwj/Intussption_classification/overlay_file_concentric")#1091


for file in sleeve:
    src_path=os.path.join("/home/gwj/Intussption_classification/data/sleeve_sign",file)
    print(src_path)
    dst_path=os.path.join("/home/gwj/Intussption_classification/overlay_file_sleeve/",file)
    print(dst_path)
    shutil.move(src_path,dst_path)
    print("success")