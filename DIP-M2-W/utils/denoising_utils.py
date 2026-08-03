import os
from .common_utils import *    
import numpy
import matplotlib.pyplot as plt
import random

def get_noisy_image(img_np, sigma):   #输入的参数分别是 np.ndarray 图像的矩阵数据  和噪声等级 sigma (人为设定)
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1     np数组的值从0-1     
        sigma: std of the noise
    """
    #real = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape) ,0 ,1).astype(np.float32)#实部加虚部
    
    #im = np.clip(np.random.normal(scale=sigma, size=img_np.shape) ,0 ,1).astype(np.float32)    #独立生成的两个 实部   虚部
   # real = img_np + sigma * np.random.standard_normal(img_np.shape)
    
    #im = sigma * np.random.standard_normal(img_np.shape)
    
    #img_noisy = np.sqrt(real**2 + im**2) 
    
   # img_noisy_np = np.clip(img_noisy, 0, 1).astype(np.float32)

#     channels, length, width = img_np.shape   #69 140 140
    
#     num_p =channels * length * width
    
#     data_noise_free = np.reshape(img_np,[-1,1]) 
    
#     real = np.reshape(sigma*np.random.standard_normal(num_p),[-1,1]) + data_noise_free
    
#     im = np.reshape(sigma*np.random.standard_normal(num_p),[-1,1])
    
#     img_noisy = np.sqrt(real**2 + im**2) 
    
#     img_noisy_np = np.reshape(img_noisy,[channels, length, width])
    
#     img_noisy_np = np.clip(img_noisy_np, 0, 1).astype(np.float32)
    
#     np.random.seed(0)
    real = img_np + sigma * np.random.standard_normal(img_np.shape)
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    
#     np.random.seed(1)   #set seed make sure the same random numbers
    im = sigma * np.random.standard_normal(img_np.shape)
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    
#     print('################Without seed#############')
#     print(np.random.standard_normal(img_np.shape)[50,40,40:45])
    img_noisy = np.sqrt(real**2 + im**2) 
    
    img_noisy_np = np.clip(img_noisy, 0, 1).astype(np.float32)

    
    return img_noisy_np   #返回加噪后的图像 (矩阵形式)

#np.random.normal(loc=0.0, scale=1.0, size=None) 均值 方差 图像大小

def findSmallest(arr):
    smallest = arr[0]      #存储最小的值
    smallest_index = 0     #存储最小元素的索引
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index
#选择排序
def selectionSort(arr):   #对数组进行排序
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))   #找出数组中最小的元素，并将其加入到新数组中
    return newArr
