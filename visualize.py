import pydicom
import matplotlib.pyplot as plt
import cv2

path = '/mnt/hdd2/qinyixin/dataset/huaxi_proj/dicom/009_SWI_Images/00064.dcm'
path = '/mnt/hdd2/qinyixin/dataset/huaxi_proj/dicom/100_MPRRange/00017.dcm'
path = '/mnt/hdd2/qinyixin/dataset/huaxi_proj/dicom/008_mIP_ImagesSW/00055.dcm'
path = '/mnt/hdd2/qinyixin/dataset/huaxi_proj/dicom/007_Pha_Images/00063.dcm'
out = './out.jpg'
ds = pydicom.read_file(path)
img = ds.pixel_array
cv2.imwrite(out, img)
