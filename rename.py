import os
import argparse
from math import sqrt
from tqdm import tqdm
import cv2
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import os.path as osp
import numpy as np
import torch
from networks.vision_transformer import SwinUnet as ViT_seg
from scipy.ndimage.interpolation import zoom
from config import get_config
import csv
import math

import matplotlib.pyplot as plt
from pylab import  *
from sympy import diff
from sympy import symbols
import heapq
from PIL import Image
from PIL import ImageFilter
import scipy.signal as signal

path="Swin-Unet/data_BVR/新增1.16"
file_list=os.listdir(path)
# # CA新增1.16的(28*30)
# i = 154
# # CA新增10.27的(19*20)
# i = 186
# BVR新增1.16的()
i = 154
for fi in file_list:
    if('CA_' in fi):
        old_name=os.path.join(path,fi)
        new_name=os.path.join(path,"image_"+str(i)+".jpg")
        os.rename(old_name,new_name)
        # lable_old_name = os.path.join("Swin-Unet/data_zEI2/labels/masks",fi[:-4]+"_mask.npy")
        # label_new_name = os.path.join("Swin-Unet/data_zEI2/labels/masks","image_"+str(i)+"_mask.npy")
        # os.rename(lable_old_name,label_new_name)
        data = {}
        data = json.loads(json.dumps(data))
        # CA_t=fi.split('CA_')[-1]
        # CA=float(CA_t[:-4])
        # data["image_"+str(i)+".jpg"]= CA
        
        with open('Swin-Unet/CA_gt.json','r', encoding="utf-8") as f:
            old_data = json.load(f)           
            old_data.update(data)
        with open('Swin-Unet/CA_gt.json','w', encoding="utf-8") as f:
            json.dump(old_data, f,indent=1)
        i += 1


        
        

    
