
import os
import random
import shutil

from PIL import Image
from torch.utils.data import Dataset

concetric=os.listdir("/home/gwj/Intussption_classification/data/concentric_circle_sign/")#5586
sleeve=os.listdir("/home/gwj/Intussption_classification/data/sleeve_sign/")#1091
print(len(concetric))

print(len(sleeve))
Intersection=list(set(concetric).intersection(set(sleeve)))
print(len(Intersection))

# for item in Intersection:
#     src_path=os.path.join("/home/gwj/Intussption_classification/data/concentric_circle_sign/",item)
#     #print(src_path)
#     dst_path=os.path.join("/home/gwj/Intussption_classification/overlay_file",item)
#     #print(dst_path)
#     shutil.move(src_path,dst_path)
#     print("success")
#overlay_file_sleeve