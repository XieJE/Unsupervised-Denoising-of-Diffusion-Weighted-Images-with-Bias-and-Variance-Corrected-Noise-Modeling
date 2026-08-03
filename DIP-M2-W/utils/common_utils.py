import torch
import torch.nn as nn
import torchvision
import sys
import scipy.io as scio
import numpy as np
from PIL import Image   #PIL python图像处理库
import PIL
import numpy as np

import matplotlib.pyplot as plt

def crop_image(img, d=32):                    #crop_image 就是对图片进行一个裁剪工作  调用PIL库中的crop函数实现
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)   #得到裁剪后图像的新尺寸 (384,256)  原来是(400,261)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),   #bbox 的数据结果拟为(snail)  [8,2,392,258]
    ]

    img_cropped = img.crop(bbox)
    return img_cropped        #返回被裁剪过后的图像

def get_params(opt_over, net, net_input, downsampler=None):          #get_params 参数的获取
    '''Returns parameters that we want to optimize over.   #返回需要优化的参数

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"            input
        net: network                                          net
        net_input: torch.Tensor that stores input `z`                   随机生成的z  就是刚开始的莫名其妙的灰色
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]   #生成了一个图像组 a list of images all of the same size 
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)   #make_grid的作用是将若干幅图像拼成一幅图像 nrow是一行放入8张图片 #参考WX九宫格
    
    return torch_grid.numpy()    

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):  #lanczos 图像采样算法
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure                             #factor     创建figure的大小
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)           ##!!!注意 先前进行了一次 transpose操作 因此 images_np (3,384,256)  max会是3
    
    #assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"  #只有单通道 或者3 通道的图像可以继续进行实验操作
                                             
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np] #np.concatenate 对数组拼接

    grid = get_image_grid(images_np, nrow)    #  grid 这里就是get_image_grid 中返回的make_grid 处理后的图像
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor)) #  figsize:指定figure的宽和高，单位为英寸；
    
    #if images_np[0].shape[0] == 1:          ##!!!注意 先前进行了一次 transpose操作 因此 images_np (3,384,256)  max会是3
    plt.imshow(grid[0], cmap='gray', interpolation=interpolation)  #如果是灰度图像 则不用进行矩阵的复原变换 transpose
    #else:
    #plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)  #将对应的坐标轴重新变换为 W H C 也就是img 的PIL模式
    
    #plt.show()                 #imshow()接收一张图像，只是画出该图，并不会立刻显示出来。   所有画完后使用plt.show()才能进行结果的显示。
    return grid

def get_image(path):                       ########有进行 图像的归一化操作 
    """Load an image and resize to a cpecific size. 
    Args: 
        path: path to image  X 固定z轴为 X   X∈(1,96)
    """
    denoised = scio.loadmat(path)
    img_np = denoised['ims_denoised'].astype(np.float32)#图像加载       b=0  一共有   0   14   28  42 56
    
    img_np = img_np[:, :, : , :]            #方便起见用3D数据进行尝试   固定z轴  取69个扩散编码方向   140*140*96*69 (z=50)

    
    b0 = [14,28,42,56]
    for z in range(img_np.shape[3]):
           if z in b0:
                img_np[:, :, :, z] = img_np[:, :, :, 50]  #去除b0
    #print(img_np.max())
    img_np = (img_np-img_np[:,:,50,:].min())/(img_np[:,:,50,:].max()-img_np[:,:,50,:].min())   #图像归一化
    #print(img_np.max())
    
    #for z in range(img_np.shape[2]):
    #       if z in b0:
    #            img_np[:, :, z] = img_np[:, :, 50]  #去除b0
    #img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())   #图像归一化
    return img_np #归一化后的矩阵img_np  <class 'numpy.ndarray'>
   

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`.""" #用噪声类型来填充张量x
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor         #张量中通道的数量
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid     #`noise`表示用noise填充张量；`meshgrid`表示np.网格
        spatial_size: spatial size of the tensor to initialize          #初始化的张量的空间大小
        noise_type: 'u' for uniform; 'n' for normal         #u 均匀分布    n 正态分布
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
                                                             #一个因子，一个噪声将乘以。基本上它是标准偏差定标器。
    """
   # if isinstance(spatial_size, int):
       # spatial_size = (spatial_size, spatial_size)
    if method == 'noise':            #'noise'
        
        shape = [1, input_depth, spatial_size[0], spatial_size[1],spatial_size[2]]
        
        net_input = torch.zeros(shape)   # 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        
        fill_noise(net_input, noise_type)
        net_input *= var          #偏差修正？  
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input     #得到初始化的用噪声填充的随机图像？？

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.       #PIL格式下 转换为 np 数组格式 图片数据
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)     #ar中存储img_PIL 矩阵

    if len(ar.shape) == 3:      #ar.shape 表示 ar矩阵的维数  (261,400,3)  高 宽 色彩通道数 140*140*69
        ar = ar.transpose(2,0,1)              # W--0 H--1 C--2  (0,1,2)轴 (三维下)    (2,0,1)---> C--2 W--0 H--1 (改变各坐标轴)
    else:
        ar = ar[None, ...]   #灰度图片则不改动

    return ar.astype(np.float32) / 255.   #变化数组类型 <class 'numpy.ndarray'> 

    #######将图片的矩阵值转成float32数据类型，然后全体除以255，其实就是将矩阵存储的整数（0，255）归一化到（0，1）的浮点数

def np_to_pil(img_np):        ##在 pil_to_np 部分np图像矩阵是进行了变化的 变化后是 C * W * H  详情见 transpose函数
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8) #np.clip 截取函数,在denoising函数中出现过  img_np是被归一化到(0,1)的浮点数 现在将他从矩阵状态复原
    
    if img_np.shape[0] == 1:
        ar = ar[0]              #如果是灰度图像则 无需进行矩阵的复原 变换  
    else:
        ar = ar.transpose(1, 2, 0) #由 C--0 W--1 H--2 ---->  (1,2,0)  W--1 H--2 C--0 变换回 PIL图像 W H C的矩阵排列方式

    return Image.fromarray(ar)   #简而言之，就是实现array到image的转换   

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.  

    From C x W x H [0..1] to  C x W x H [0..1] 
    '''
    return torch.from_numpy(img_np)[None, :]  ###  把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]   #张量形式转化成np 数组形式


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'                   #梯度下降算法
        parameters: list of Tensors to optimize over            #待优化的张量列表 由get_param获得
        closure: function, that returns loss variable           #清除梯度，计算并返回损失
        LR: learning rate                              #学习率             
        num_iter: number of iterations                     #迭代数量
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)  #优化器对象Optimizer   学习率默认0.001
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')  
        optimizer = torch.optim.Adam(parameters, lr=LR)   #优化器对象Optimizer   学习率lr传递LR=0.01
        
        for j in range(num_iter):     #此处开始3000此次迭代
            optimizer.zero_grad()        #梯度置零
            closure()                #写的一个损失函数一样的东西 
            optimizer.step()           #optimizer.step()   更新权重参数      ... 进行下一循环
    elif optimizer_type == 'SGD':
        print('Starting optimization with SGD')  
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        for j in range(num_iter):     
            optimizer.zero_grad()        
            closure()               
            optimizer.step()                  
    else:
        assert False