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


for filename in os.listdir('Swin-Unet/data_zEI_2'):
    data = {}
    data = json.loads(json.dumps(data))
    F = float(filename[9:13])
    G = float(filename[16:-4])
    ZEI = F/G
    new_data = {
        "F":F,
        'G':G,
        'ZEI':ZEI
    }
    data[filename]= new_data
    with open('Swin-Unet/zEI_gt.json','r', encoding="utf-8") as f:
        old_data = json.load(f)           
        old_data.update(data)
    with open('Swin-Unet/zEI_gt.json','w', encoding="utf-8") as f:
        json.dump(old_data, f,indent=1)