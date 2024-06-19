import argparse
from math import sqrt
from tqdm import tqdm
import cv2
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from scipy import ndimage
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Huaxi', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--model_path', default = 'output_4_25/epoch_299.pth',type=str)  
parser.add_argument('--output_dir', default = 'output_4_25',type=str, help='output dir')                   
parser.add_argument('--data_path', default = '2024_4_25_dataset/images',type=str, help='data_path image dir or an single image')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )                  
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

cfgs = get_config(args)

lines = []

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    img = img.astype(np.uint8)
    return cv2.warpAffine(img, M, (nW, nH))

def visualization(output,img_name):
    #outPutImg 是npy数组 *255是因为数组中的元素非零即一
    outputImg = Image.fromarray(output*255/2)
    #"L"代表将图片转化为灰度图
    outputImg = outputImg.convert('L')
    path = osp.join('result_425','v_' + img_name)
    outputImg.save(path)  
    return path

def task4_filter(output): 
    # 均值滤波
    kernel = np.ones((2,2)) * 1 / (2 * 2)
    img = signal.convolve2d(output, kernel, mode = 'same')
    img = np.rint(img)
    # 中值滤波
    img = signal.medfilt(output,(3,3))

    return img 
                                                                 
def jiuzheng(file_path):
    img = cv2.imread(file_path)
    ori_path=osp.join('result_zEI_zoomori'+file_path[15:])
    ori_img=cv2.imread(ori_path)
    # 二值化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    #霍夫变换，摘自https://blog.csdn.net/feilong_csdn/article/details/81586322
    lines = cv2.HoughLines(edges,1,np.pi/180,0)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    if x1 == x2 or y1 == y2:
        t=0
    else:
        t = float(y2-y1)/(x2-x1)
    # 得到角度后
    rotate_angle = math.degrees(math.atan(t))
    print(file_path+"    angle:  "+ str(rotate_angle))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    # 图像根据角度进行校正
    rotate_img = ndimage.rotate(ori_img, rotate_angle)

    # 输出图像
    imageio.imwrite(ori_path, rotate_img)



def Rule_based(output, img_name, angle = 0):
    output = np.squeeze(output,axis=0) # (224, 224)
    # output = task4_filter(output)
    v_path = visualization(output,img_name)

    #..............
    #倾斜矫正
    
    # 还原到原图
    # 可视化zoom到原始尺寸
    im = cv2.imread(v_path, 1)
    zoom_im = zoom(im, ((224/im.shape[0])*zoom_x, (224/im.shape[1])*zoom_y,1), order=3)
    zoom_path=osp.join('result_425','v_' + img_name)       
    cv2.imwrite(zoom_path, zoom_im)
    zoomori_path=osp.join('result_425_zoomori','v_' + img_name)
    im = cv2.imread(zoomori_path, 1)              
    dst=cv2.addWeighted(im,0.6,zoom_im,0.4,0)
    cv2.imwrite(zoom_path, dst)


def preprocess(img_path,img_name):
    image = cv2.imread(img_path, 0)
    # ...
    # img_for_jiuzheng=cv2.imread(img_path)
    # h,w = image.shape
    # image = image[50:h-50, 100:w - 100]
    new_h, new_w = image.shape
    tmp = np.nonzero(image)
    hmin,hmax = np.min(tmp[0]), np.max(tmp[0])
    wmin,wmax = np.min(tmp[1]), np.max(tmp[1])

    hmax = hmax + 10 if hmax + 10 <= new_h else new_h
    hmin = hmin - 10 if hmin - 10 >= 0 else 0
    wmax = wmax + 10 if wmax + 10 <= new_w else new_w
    wmin = wmin - 10 if wmin - 10 >= 0 else 0

    image = image[hmin:hmax, wmin:wmax]
    # 还原需要
    # img_for_jiuzheng=img_for_jiuzheng[hmin:hmax,int((wmax-wmin)/2):wmax,:]
    # for_path=osp.join('for_zEI_zoomori','v_' + img_name) 
    # cv2.imwrite(for_path, img_for_jiuzheng)
    # jiuzheng(for_path)
    zoomori_path=osp.join('result_425_zoomori','v_' + img_name)       
    cv2.imwrite(zoomori_path, image)
    x, y = image.shape
    if x != 224 or y != 224:
        #为了之后可视化还原的参数
        global zoom_x
        global zoom_y
        zoom_x = x/224
        zoom_y = y/224
        image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

    return image
zoom_x=0
zoom_y=0
def main():
    
    if osp.isfile(args.data_path):
        img_paths = [args.data_path]
    elif osp.isdir(args.data_path):
        img_paths = os.listdir(args.data_path)
        img_paths.sort()
        img_paths = [osp.join(args.data_path, i) for i in img_paths]
    
    model = ViT_seg(cfgs, img_size=cfgs.DATA.IMG_SIZE, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load('output_4_25/epoch_299.pth'))
    model.eval()
    for path in tqdm(img_paths):
        img_name = osp.basename(path)
        # img_name = "".join(list(filter(str.isdigit, img_name)))
        
        img = preprocess(path,img_name)
        img = img.unsqueeze(0).cuda()
        outputs = model(img)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.data.cpu().numpy()
        score = Rule_based(outputs, img_name)
    



lines.append(["image_name", "ground_truth", "predicted","error"])
main()  
with open('statistics.csv','w') as f:
    w = csv.writer(f)
    w.writerows(lines)

###
# boundary : 2
# center : 1