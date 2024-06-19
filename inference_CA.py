import argparse
from math import sqrt
from tqdm import tqdm
import cv2
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Huaxi', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--model_path', default = 'Swin-Unet/output/epoch_299.pth',type=str)  
parser.add_argument('--output_dir', default = 'Swin-Unet/output_ourT2',type=str, help='output dir')                   
parser.add_argument('--data_path', default = 'Swin-Unet/data_CA/images',type=str, help='data_path image dir or an single image')
parser.add_argument('--cfg', type=str, default='Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
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
                    default=24, help='batch_size per gpu')
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

def compute_angle(yl, xl, yc, xc, yr, xr):
    arc1 = (yc - yl) / (xc - xl)
    arc2 = (yc - yr) / (xr - xc)
    angle = math.pi - math.atan(arc1) - math.atan(arc2)
    angle = angle * 180 / math.pi
    return angle

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

def Rule_based(output, img_name, angle = 0):
    
    output = np.squeeze(output,axis=0) # (224, 224)
    v_path = visualization(output,img_name)
    
    for an in range(-angle, angle + 1):
        
        img = rotate_image(output, an)
        # HW轴互换
        img = np.transpose(img)
        tmp = np.nonzero(img)
        w_axis = list(tmp[0])
        h_axis = tmp[1]
        w_index = []
        for i in w_axis:
            w_index.append(w_axis.index(i))
        w_index = list(set(w_index))
        w_index = np.array(w_index)
        w_index = sorted(w_index,reverse=False)
       
        w_new = tmp[0][w_index]
        h_new = h_axis[w_index]

        # 多项式拟合的h坐标
        func = np.polyfit(w_new,h_new,5)
        func = np.poly1d(func)
        h_func = func(w_new)

        # 求导找5点
        x = symbols("x")
        Derivatives_ori = []
        for i in w_new:
            Derivatives_ori.append(diff(func(x),x).subs(x,i))
        Derivatives = list(abs(np.array(Derivatives_ori)))
        arr_min = heapq.nsmallest(5,Derivatives)#获取最小的五个值并排序
        index_min = list(map(Derivatives.index,arr_min))#获取最小的五个值的下标
        index_min = sorted(np.array(index_min),reverse=False)
        
        # 方法一(切线)
        index_left = round((index_min[0] + index_min[2]) / 2)
        index_right = round((index_min[4] + index_min[2]) / 2)
        left_interaval = index_min[2] - index_left
        right_interval = index_right - index_min[2]
        angle1 = 0
        interval = round(1 / 2 * min(left_interaval, right_interval))
        for p in range(0, 1):
            # 斜率
            left = Derivatives_ori[max(index_left + p ,0)]
            right = Derivatives_ori[min(index_right - p,len(w_new) - 1)]
            h_func_left = h_new[index_left] + (w_new[:index_min[2] + 1] - w_new[index_left]) * left
            h_func_right = h_new[index_right] + (w_new[index_min[2]:] - w_new[index_right]) * right
            # 向量
            left = np.array([w_new[0] - w_new[index_min[2]],h_func_left[0] - h_func_left[-1]])
            right = np.array([w_new[-1] - w_new[index_min[2]],h_func_right[-1] - h_func_right[0]])
            l_left = math.sqrt(left.dot(left))
            l_right = math.sqrt(right.dot(right))
            dian = left.dot(right)
            cos_ = dian / (l_left * l_right)
            angle1 = math.acos(cos_)
            angle1 = angle1 / math.pi * 180

            # 用向量计算的

        # angle1 = angle1 / 1

        # 方法2：蝴蝶装上半部分最低点和左右中间点连线夹角
        # 获取h最高的点-1
        # h_new_part = list(np.array(h_new[index_min[0]:index_min[4]]))
        # arr_max = heapq.nlargest(5,h_new_part)#获取最小的五个值并排序  
        # index_max = list(map(h_new_part.index,arr_max))
        # index_max = sorted(np.array(index_max),reverse=False)
        # index_max = index_max[2] + index_min[0] + 1
        # 获取h最高的点-2
        index_max = index_min[2]

        # 向量
        left = np.array([w_new[index_left] - w_new[index_max],h_new[index_left] - h_new[index_max]])
        right = np.array([w_new[index_right] - w_new[index_max],h_new[index_right] - h_new[index_max]])
        l_left = math.sqrt(left.dot(left))
        l_right = math.sqrt(right.dot(right))
        dian = left.dot(right)
        cos_ = dian / (l_left * l_right)
        angle2 = math.acos(cos_)
        angle2 = angle2 / math.pi * 180
        angle = (angle1 + angle2) / 2
        # 方法2可视化数据
        k_left = (h_new[index_left] - h_new[index_max]) / (w_new[index_left] - w_new[index_max])
        k_right = (h_new[index_right] - h_new[index_max]) / (w_new[index_right] - w_new[index_max])
        h2_func_left = h_new[index_max] + (w_new[:index_max + 1] - w_new[index_max]) * k_left
        h2_func_right = h_new[index_max] + (w_new[index_max:] - w_new[index_max]) * k_right

        # 可视化
        im = array(Image.open(v_path))
        imshow(im,cmap='gray')   
        plot((w_new[:index_min[2] + 1]),h_func_left)
        plot((w_new[index_min[2]:]),h_func_right)
        # plot((w_new[:index_max + 1]),h2_func_left)
        # plot((w_new[index_max:]),h2_func_right)    
        plt.axis('off') # 去坐标轴
        plt.xticks([]) # 去刻度
        plt.yticks([]) # 去刻度 
        savefig(v_path,bbox_inches='tight',pad_inches = -0.01)
        plt.close()   
        #可视化zoom到原始尺寸
        im = cv2.imread(v_path, 1)
        zoom_im = zoom(im, ((224/im.shape[0])*zoom_x, (224/im.shape[1])*zoom_y,1), order=3)
        zoom_path=osp.join('result_ca_zoom','v_' + img_name)       
        cv2.imwrite(zoom_path, zoom_im)
        zoomori_path=osp.join('result_ca_zoomori','v_' + img_name)
        im = cv2.imread(zoomori_path, 1)              
        dst=cv2.addWeighted(im,0.6,zoom_im,0.4,0)
        cv2.imwrite(zoom_path, dst)

    return angle

def visualization(output,img_name):
    #outPutImg 是npy数组 *255是因为数组中的元素非零即一
    outputImg = Image.fromarray(output*255.0)
    #"L"代表将图片转化为灰度图
    outputImg = outputImg.convert('L')
    path = osp.join('result_CA','v_' + img_name)
    outputImg.save(path)  
    return path

def Regression_based(output, img_name, angle = 0):
    
    output = np.squeeze(output,axis=0) # (224, 224)
    v_path = visualization(output,img_name)
    
    if img_name == 'image_121.jpg':
        print("hold on")
    for an in range(-angle, angle + 1):
        
        img = rotate_image(output, an)
        # HW轴互换
        img = np.transpose(img)
        tmp = np.nonzero(img)
        w_axis = list(tmp[0])
        h_axis = tmp[1]
        w_index = []
        for i in w_axis:
            w_index.append(w_axis.index(i))
        w_index = list(set(w_index))
        w_index = np.array(w_index)
        w_index = sorted(w_index,reverse=False)      
        w_new = tmp[0][w_index]
        h_new = h_axis[w_index]

        # 多项式拟合的h坐标
        func = np.polyfit(w_new,h_new,5)
        func = np.poly1d(func)

        # 求导找5点
        x = symbols("x")
        Derivatives_ori = []
        for i in w_new:
            Derivatives_ori.append(diff(func(x),x).subs(x,i))
        Derivatives = list(abs(np.array(Derivatives_ori)))
        arr_min = heapq.nsmallest(5,Derivatives)#获取最小的五个值并排序
        index_min = list(map(Derivatives.index,arr_min))#获取最小的五个值的下标
        index_min = sorted(np.array(index_min),reverse=False)

        #获取h最高的点
        h_new_part = list(np.array(h_new[index_min[0]:index_min[4]]))
        arr_max = heapq.nlargest(5,h_new_part)#获取最小的五个值并排序  
        index_max = list(map(h_new_part.index,arr_max))
        index_max = sorted(np.array(index_max),reverse=False)
        index_max = index_max + index_min[2]

        #分左右两段，直线拟合       
        func_left = np.polyfit(w_new[index_min[0]:index_max[0]],h_new[index_min[0]:index_max[0]],1)
        func_right = np.polyfit(w_new[index_max[0]:index_min[4]],h_new[index_max[0]:index_min[4]],1)
        left = func_left[0]
        right = func_right[0]
        func_left = np.poly1d(func_left)
        func_right = np.poly1d(func_right)
        angle = math.atan(abs((right - left) / (1 + (right * left))))
        angle = 180 - angle/math.pi*180

        # 可视化
        im = array(Image.open(v_path))
        imshow(im,cmap='gray')
        h_func_left = func_left(w_new[0:index_max[0] + 1])
        h_func_right = func_right(w_new[index_max[0]:])      
        plot((w_new[0:index_max[0] + 1]),h_func_left)
        plot((w_new[index_max[0]:]),h_func_right)
        # plt.xlim(0,224)
        # plt.ylim(0,224)
        savefig(v_path)#保存图片
        plt.close()
    return angle

def preprocess(img_path,img_name):
    image = cv2.imread(img_path, 0)
    h,w = image.shape
    image = image[:, w//2:]
    h,w = image.shape
    image = image[50:h-50, 100:w - 100]
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
    zoomori_path=osp.join('result_ca_zoomori','v_' + img_name)       
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
        img_paths = [osp.join(args.data_path, i) for i in img_paths]
    
    model = ViT_seg(cfgs, img_size=cfgs.DATA.IMG_SIZE, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load('Swin-Unet/output/epoch_299.pth'))
    model.eval()
    gt_scores = json.load(open('Swin-Unet/CA_gt.json'))
    error = 0
    bad_case = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    threshold = 90
    for path in tqdm(img_paths):
        img_name = osp.basename(path)
        gt_score = gt_scores[img_name]

        img = preprocess(path,img_name)
        img = img.unsqueeze(0).cuda()
        outputs = model(img)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.data.cpu().numpy()
        # 可视化

        angle = Rule_based(outputs, img_name)

        if angle < threshold and gt_score < 90:
            true_positive += 1
            print("11: " + img_name)
        elif angle >= threshold and gt_score < 90:
            false_positive += 1
            print("12: " + img_name)
        elif angle < threshold and gt_score >= 90:
            false_negative += 1
            print('21: ' + img_name)
        else:
            true_negative += 1
        error += abs(angle - gt_score) 
        if(abs(angle - gt_score) > 8):
            print("bad casez: " + img_name)
            print('a')

        #vertical_image(outputs)
        lines.append([img_name, gt_score, angle, abs(angle-gt_score)])
    recall = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)
    # f1 = 2*recall*precision / (recall + precision)
    print("Avg error : {}".format(error/(len(img_paths)-1)))
    print(len(img_paths))
    # print("召回率：{}".format(recall) + "  精确率： {}".format(precision) + "  f1: {}".format(f1))


lines.append(["image_name", "ground_truth", "predicted","error"])
main()
with open('statistics.csv','w') as f:
    w = csv.writer(f)
    w.writerows(lines)

###
# boundary : 1
# center : 2