# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
# import cv2
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(256, 256)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    
    return image, label

class CT_norm(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(128,128)):
        super(CT_norm, self).__init__()
        self.img_lst,self.label = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""
        compose = [
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),  
            ] if isTraining==True else [
                transforms.ToTensor(),  
            ]
        self.transform = transforms.Compose(compose)
        
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3
    
    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        label = self.label[index]
        simple_transform = transforms.ToTensor() 
        img = Image.open(imgPath)
        # gt = Image.open(gtPath).convert("L").resize(self.scale_size, Image.BICUBIC)
        
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        img = img.resize(self.scale_size, Image.BICUBIC)
        # gt = gt.resize(self.scale_size, Image.BICUBIC)
        # gt = np.array(gt)
        # gt[gt >= 128] = 255
        # gt[gt < 128] = 0
        # gt = Image.fromarray(gt)

        # gt = gt.convert('L')
        
        # if self.isTraining:
        #     # augumentation
        #     rotate = 10
        #     angel = random.randint(-rotate, rotate)
        #     img = img.rotate(angel)
        #     # gt = gt.rotate(angel)
        
        img = self.transform(img)
        # gt = simple_transform(gt)

        
        return img, label
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)
    
    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir = os.path.join(root + "/train")
            # gt_dir = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/ponu_ad/CT_detection_segmentation'
        else:
            file_dir = os.path.join(root + "/test_new")
            # gt_dir = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/ponu_ad/CT_detection_segmentation'
        
        # img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        # gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        img_lst = []
        # gt_lst = []
        label = []
        file_list = os.listdir(file_dir)
        file_list.sort()
        for item in file_list:
           file_path1 = os.path.join(file_dir,item)
           file_list1 = os.listdir(file_path1)
           file_list1.sort()
           for item2 in file_list1:
                 image_path = os.path.join(file_path1,item2)
                 img_lst.append(image_path)
                #  gt_path = os.path.join(gt_dir, item, item1,item2)
                #  gt_lst.append(gt_path)
                 if item == 'normal':
                     label.append(int(0))
                 else:
                     label.append(int(1))
                #  if item == 'BP':
                #      label.append(int(1))
                #  if item == 'cancer':
                #      label.append(int(2))
                #  if item == 'COVID-19':
                #      label.append(int(3))
                #  if item == 'cryptococcosis':
                #      label.append(int(4))
                #  if item == 'fungal':
                #      label.append(int(5))
                #  if item == 'mycoplasma':
                #      label.append(int(6))
                #  if item == 'others':
                #      label.append(int(7))



        
        return img_lst, label
    
    def getFileName(self):
        return self.img_lst
