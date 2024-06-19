import cv2
import os
import os.path as osp
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import CenterCrop


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample




class Huaxi_RandomGenerator(object):
    def __init__(self, output_size, args):
        self.output_size = output_size
        self.task = args.task
        self.num_classes = args.num_classes
        if self.task == 2:
            self.crop = CenterCrop(224)

    def __call__(self, image, label):
        h,w = image.shape
        
        if self.task == 2:
            image = image[:, w//2:]
            label = label[:, w//2:]
        
        h,w = image.shape
        image = image[50:h-50, 100:w - 100]
        label = label[50:h-50, 100:w - 100]
        new_h, new_w = image.shape
        #得到非零元素的位置，输出两个数组
        tmp = np.nonzero(image)
        hmin,hmax = np.min(tmp[0]), np.max(tmp[0])
        wmin,wmax = np.min(tmp[1]), np.max(tmp[1])

        #裁剪出颅骨图
        hmax = hmax + 10 if hmax + 10 <= new_h else new_h
        hmin = hmin - 10 if hmin - 10 >= 0 else 0
        wmax = wmax + 10 if wmax + 10 <= new_w else new_w
        wmin = wmin - 10 if wmin - 10 >= 0 else 0

        image = image[hmin:hmax, wmin:wmax]
        label = label[hmin:hmax, wmin:wmax]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        x, y = image.shape
        if self.task == 2:
            square = min(x, y)
            x = square
            y = square
            image = image[:square, :square]
            label = label[:square, :square]
        
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        #尺寸是num_classesx224x224
        target = torch.zeros((self.num_classes,label.shape[0],label.shape[1]))

        label = torch.from_numpy(label).to(torch.int64).unsqueeze(0)

        target.scatter_(dim = 0,index = label,value=1)
        
        #sample = {'image': image, 'label': label.long()}
        
        return image, target.long()



class Huaxi_dataset(Dataset):
    def __init__(self, split, base_dir, img_dir, label_dir, transform):
        super().__init__()
        self.split = split
        self.transform = transform
        f = open(osp.join(base_dir, split + '.txt')).readlines()
        self.idx_list = [i.strip() for i in f]
        self.img_dir = osp.join(base_dir, img_dir)
        self.label_dir = osp.join(base_dir, label_dir)
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, index):
        img_id = self.idx_list[index]
        img_path = osp.join(self.img_dir, "image_%s.jpg"%str(img_id))
        img = cv2.imread(img_path,0)

        label_path = osp.join(self.label_dir, "image_%s_mask.npy"%str(img_id))
        label = np.load(label_path)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label