import argparse
from math import sqrt
from tqdm import tqdm
import cv2
from copy import deepcopy
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Huaxi', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--model_path', default = 'Swin-Unet/output/task1_best.pth',type=str)  
parser.add_argument('--output_dir', default = 'result',type=str, help='output dir')                   
parser.add_argument('--data_path', default = 'Swin-Unet/data_EI/images',type=str, help='data_path image dir or an single image')
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

def scan_image(output, img_name, angle = 0):
    
    output = np.squeeze(output,axis=0) # (224, 224)
    v_path = visualization(output,img_name)

    data = []
    for an in range(-angle, angle + 1):
        
        img = rotate_image(output, an)
        tmp = np.nonzero(img)
        hmin, hmax = np.min(tmp[0]), np.max(tmp[0])
        wmin, wmax = np.min(tmp[1]), np.max(tmp[1])
        image = img[hmin:hmax + 1, wmin: wmax + 1]
        
        b_image = deepcopy(image)
        c_image = deepcopy(image)
        h,w = image.shape
        # scan boundary:
        for row in range(h):
            tmp = np.nonzero(b_image[row])[0]
            if len(tmp) ==0 :
                continue
            w_min, w_max = np.min(tmp), np.max(tmp)
            b_image[row, w_min : w_max +1] += 1  # boundary:2, center:3, head:1
        
        b_image[b_image==2] = 0
        b_image[b_image==3] = 0
        
        tmp = np.nonzero(b_image)
        
        # 找最外层的点
        #b_minw, b_maxw = np.min(tmp[1]), np.max(tmp[1])
        #b_minh = np.median(np.nonzero(b_image[:, b_minw])[0])
        #b_maxh = np.median(np.nonzero(b_image[:, b_maxw])[0])

        # 横线扫描
        
        axis_x = set(tmp[0].tolist())
        b_maxh = 0
        b_minh = 1000
        b_maxw = 0
        b_minw = 1000
        max_bound = 0
        for x in axis_x:
            row = b_image[x]
            tmp_row = np.nonzero(row)[0]
            interval = np.max(tmp_row) - np.min(tmp_row)
            if interval > max_bound:
                max_bound = interval
                b_minw, b_maxw = np.min(tmp_row), np.max(tmp_row)
                b_minh = x
                b_maxh = x

        
        # scan center
        c_image[c_image==1] = 0
        # find crop
        tmp = np.nonzero(c_image)
        x_axis = list(set(tmp[0]))
        crop_x = (max(x_axis) - min(x_axis))//3 + min(x_axis)

        c_image = c_image[0:crop_x, :] # crop half
        tmp = np.nonzero(c_image)
        if tmp[0].shape[0] ==0:
            return 0
        #找最外层的点
        #c_minw, c_maxw = np.min(tmp[1]), np.max(tmp[1])
        #c_minh = np.median(np.nonzero(c_image[:, c_minw])[0])
        #c_maxh = np.median(np.nonzero(c_image[:, c_maxw])[0])

        # scan center
        
        axis_x = set(tmp[0].tolist())
        c_maxh = 0
        c_minh = 1000
        c_maxw = 0
        c_minw = 1000
        max_bound = 0
        for x in axis_x:
            row = c_image[x]
            tmp_row = np.nonzero(row)[0]
            interval = np.max(tmp_row) - np.min(tmp_row)
            if interval > max_bound:
                max_bound = interval
                c_minw, c_maxw = np.min(tmp_row), np.max(tmp_row)
                c_minh = x
                c_maxh = x
        
        image = np.expand_dims(image, axis = 2) * 50
        image = image.astype(np.uint8)
        cv2.line(image,(b_minw,int(b_minh)),(b_maxw,int(b_maxh)),(255,0,0),2)
        cv2.line(image,(c_minw,int(c_minh)),(c_maxw,int(c_maxh)),(255,0,0),2)
        #padding
        pad_w,pad_h=img.shape
        image_pad = np.zeros((pad_w,pad_h,1))   
        image_pad[hmin:hmax + 1, wmin: wmax + 1] = image    #再把原矩阵放到相应位置

        # arr_by = np.full(b_maxw-b_minw,b_minh)
        # arr_bx = array(range(b_minw, b_maxw))
        # arr_cy = np.full(c_maxw-c_minw,c_minh)
        # arr_cx = array(range(c_minw, c_maxw))
        # im = array(Image.open(v_path))
        # imshow(im,cmap='gray')        
        # plot(arr_bx,arr_by) 
        # plot(arr_cx,arr_cy) 

        # plt.axis('off') # 去坐标轴
        # plt.xticks([]) # 去刻度
        # plt.yticks([]) # 去刻度 
        # savefig(v_path,bbox_inches='tight',pad_inches = -0.01)
        # plt.close()

        # yuanshi
        cv2.imwrite(v_path,image_pad)
        # 可视化zoom到原始尺寸
        im = cv2.imread(v_path, 1)
        zoom_im = zoom(im, ((224/im.shape[0])*zoom_x, (224/im.shape[1])*zoom_y,1), order=3)
        zoom_path=osp.join('result_EI_zoom','v_' + img_name)       
        cv2.imwrite(zoom_path, zoom_im)
        zoomori_path=osp.join('result_EI_zoomori','v_' + img_name)
        im = cv2.imread(zoomori_path, 1)              
        dst=cv2.addWeighted(im,0.6,zoom_im,0.4,0)
        cv2.imwrite(zoom_path, dst)

        A = sqrt(pow(c_maxh - c_minh, 2) + pow(c_maxw - c_minw,2))
        B = sqrt(pow(b_maxh - b_minh, 2) + pow(b_maxw - b_minw, 2))
        EI = A/B
        data.append(EI)
    EI = (sum(data) - max(data) - min(data)) / (len(data) - 2)
    #EI = max(data)
    
    return EI


def preprocess(img_path,img_name):
    image = cv2.imread(img_path, 0)
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
    zoomori_path=osp.join('result_EI_zoomori','v_' + img_name)       
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

def visualization(output,img_name):
    #outPutImg 是npy数组 *255是因为数组中的元素非零即一
    outputImg = Image.fromarray(output*100.0)
    #"L"代表将图片转化为灰度图
    outputImg = outputImg.convert('L')
    path = osp.join('result_EI','v_' + img_name)
    outputImg.save(path)  
    return path

zoom_x=0
zoom_y=0
def main():
    
    if osp.isfile(args.data_path):
        img_paths = [args.data_path]
    elif osp.isdir(args.data_path):
        img_paths = os.listdir(args.data_path)
        img_paths = [osp.join(args.data_path, i) for i in img_paths]
    
    model = ViT_seg(cfgs, img_size=cfgs.DATA.IMG_SIZE, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load('Swin-Unet/output/task1_best.pth'))
    model.eval()
    gt_scores = json.load(open('Swin-Unet/EI_gt.json'))
    error = 0
    bad_case = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    threshold = 0.3
    for path in tqdm(img_paths):
        img_name = osp.basename(path)
        gt_score = float(gt_scores[img_name][0])

        img = preprocess(path,img_name)
        img = img.unsqueeze(0).cuda()
        outputs = model(img)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.data.cpu().numpy()

        score = scan_image(outputs, img_name)
        if score == 0:
            bad_case.append(img_name)
            continue
        elif abs(score - gt_score) > 0.3:
            bad_case.append(img_name)
    
        if score >= threshold and gt_score >= 0.3:                           
            true_positive += 1
        elif score <threshold and gt_score >=0.3:
            false_positive += 1
        elif score >=threshold and gt_score <0.3:
            false_negative += 1
        else:
            true_negative += 1
        error += abs(score - gt_score) 
        
        
        #vertical_image(outputs)
        lines.append([img_name, gt_score, score, abs(score-gt_score)])
    recall = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)
    f1 = 2*recall*precision / (recall + precision)
    print("Avg error : {}".format(error/(len(img_paths)-1)))
    print("recall : {}".format(recall))
    print("precision : {}".format(precision))


lines.append(["image_name", "ground_truth", "predicted","error"])
main()
with open('statistics.csv','w') as f:
    w = csv.writer(f)
    w.writerows(lines)

###
# boundary : 1
# center : 2