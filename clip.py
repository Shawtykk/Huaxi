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

path = "Swin-Unet/data_BVR/新增1.16"

# path = "Swin-Unet/data_zEI/labels"
for filename in os.listdir(path):
    old_name = os.path.join(path, filename)
    new_name = os.path.join(path, "E_" + filename.split("E_")[1])
    os.rename(old_name,new_name)