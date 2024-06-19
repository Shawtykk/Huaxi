import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import onnxruntime as ort
from sklearn import svm
from tqdm import tqdm
from scipy.ndimage import zoom, rotate
import torch
import math

def load_dicom_folder(folder_path):
    # 读取文件夹中所有的DICOM文件
    files = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image_volume = np.stack([file.pixel_array for file in files])
    return image_volume, files

### 读取数据 #

if __name__ == "__main__":
    dicom_dir = "原始图像11-45/原始图像11-15/ST000011/SE000000"
    image, files = load_dicom_folder(dicom_dir)
    img_name = dicom_dir.split("/")[-2]

    ### 从10~100层，每间隔3层取一个值
    slices = torch.stack([preprocess(image[:, i, :]) for i in range(10, 100, 3)])
    ori_images = [image[:, i, :] for i in range(10, 100, 3)]
    print(f"shape of slices: {slices.shape}")
