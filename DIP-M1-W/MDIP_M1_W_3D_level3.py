from __future__ import print_function 
import numpy as np
from models import * 
import torch
import torch.optim
from utils.denoising_utils import * 
from PIL import Image   
import PIL
import matplotlib.pyplot as plt
import os
from dipy.io.image import load_nifti


torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = True 
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

show_every = 40


# load data
mask,_ = load_nifti('/home/chenyunwei/DIP_data/generate_data/30/mask.nii.gz')
mask = mask[:,:,10:]
mask_np = mask.astype(np.float32)
# 裁剪图片，减小计算
img_np_mask = mask_np
# 检查shape
print(img_np_mask.shape)
# load noise free image
img_np,_ = load_nifti('/home/chenyunwei/DIP_data/generate_data/30/dwi_reference.nii.gz')
img_np = img_np[:,:,10:,:]
img_np = img_np.astype(np.float32)
# 噪声水平 可改
noise_level = 3

# 裁剪图片，减小计算

# 检查shape 
print(img_np.shape)
# 转置数据，符合网络输入
img_np = img_np.transpose(3,0,1,2)


# 加载噪声数据
img_noisy_np,_ = load_nifti('/home/chenyunwei/DIP_data/generate_data/30/dwi_level_' + str(noise_level) +'.nii.gz')
img_noisy_np = img_noisy_np.astype(np.float32)
img_noisy_np = img_noisy_np[:,:,10:]
img_noisy_np = img_noisy_np.transpose(3,0,1,2)
print(img_noisy_np.shape)
sigma_ = noise_level  /100

# mask 后的噪声数据
img_noisy_np_nonskull = img_noisy_np * img_np_mask
print(img_noisy_np_nonskull.shape)
# mask 后的无噪数据
img_np_nonskull = img_np * img_np_mask


#设置网络
INPUT = 'noise' 
OPT_OVER = 'net'
#填充方式 反射
pad = 'reflection'
i = 0


# 学习率
LR = 0.01

# 优化算法
OPTIMIZER='adam'
#最大迭代次数
num_iter = 200000
#输入通道数 
input_depth = 31
# 网络参数初始化
net = get_net(input_depth, 'skip', pad,skip_n33d=32,skip_n33u=32,skip_n11=4,num_scales=4,upsample_mode='trilinear',n_channels=31).type(dtype)




# 网络输入 5D 随机张量 并裁剪加速计算
net_input,_ = load_nifti('/home/chenyunwei/DIP_data/generate_data/30/noisy_input.nii.gz')
net_input = net_input[:,:,10:,:]
net_input = net_input.transpose(3,0,1,2)
net_input = np.expand_dims(net_input,axis=0)
net_input = net_input.astype(np.float32)
net_input = torch.from_numpy(net_input).cuda() 
# 检查数据是否位于GPU，shape
print(net_input.device)
print(type(net_input))
print(net_input.shape)
# loss 均方误差 mse
mse = torch.nn.MSELoss().type(dtype)
# 将有噪数据和无噪数据转换为torch tensor
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype) # 把数组转换成张量，且二者共享内存 img_noisy_np内存储的是加噪后的图像 

# normal
psrn_noisy_list = []  
psrn_out_list = []  
rmse_out_list = []
total_loss_list = [] 

# # 断点重新训练
# bp = True
# ii = np.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/iterations/test/i.npy')
# i = np.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/iterations/test/i.npy')
# i = i-(i%40)
# LR=0.01*0.9**(i//2000)

# trained_model_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/trained_model/test/epoch_' + str(i) +'.pt'
# trained_model_name_last = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/trained_model/test/epoch_' + str(i-1) +'.pt'
# net.load_state_dict(torch.load(trained_model_name_last))
# out_bp = net(net_input).data
# net.load_state_dict(torch.load(trained_model_name))

# psrn_out_list = list(np.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/psnr_test.npy'))
# total_loss_list = list(np.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/loss_test.npy'))
# rmse_out_list = list(np.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/rmse_test.npy'))

# for k in range(i,ii+1):
#     del total_loss_list[i]
#     del psrn_out_list[i]
#     del rmse_out_list[i]

net_input_saved = net_input.detach().clone()   
noise = net_input.detach().clone()              
# print(type(net_input))
out_avg = None
last_net = None
psrn_noisy_last = 0              
total_loss_last = 9999
psrn_gt_last = 0
t=0

mask_num_npy = np.sum(img_np_mask)
mask_num = np.sum(img_np_mask)
mask_num=128*160*111/mask_num
mask_num = torch.tensor(mask_num,dtype=torch.float32,device='cuda',requires_grad=False)  
mask_num_sqrt = torch.sqrt(mask_num)

last_i=0
def closure():   #######！！！！！#####
    
    global last_i,total_loss_last,num_iter,t,psrn_gt_last,LR,i, out_avg, psrn_noisy_last, last_net, net_input, psrn_noisy_list, psrn_gt_list,total_loss_list   #去噪结果PSNR  
    out = net(net_input)    # <class 'torch.Tensor'>
# 上一次迭代的输出
    alpha1 = (out / (2*sigma_))**2   
    pi = torch.tensor(np.pi,dtype=torch.float32,device='cuda',requires_grad=False) 
    SNR = out.data/sigma_
    # ep = (SNR**2+2)-(pi/8)*(((2+SNR**2)*torch.special.i0e((SNR**2)/4)+(SNR**2)*torch.special.i1e((SNR**2)/4)))**2
    ep = (2 + SNR**2) - (pi/8)*(((2+SNR**2)*torch.special.i0e((SNR**2)/4) + (SNR**2)*torch.special.i1e((SNR**2)/4))**2)
    M1_weight = sigma_*torch.sqrt(ep)
    temp = torch.sqrt((pi*(sigma_**2))/2)*((1+2*alpha1)*torch.special.i0e(alpha1) + 2*alpha1*torch.special.i1e(alpha1))
    total_loss = mse(temp/M1_weight, img_noisy_torch/M1_weight)
    total_loss.backward() 
    total_loss_ = total_loss.data.cpu().numpy()
    total_loss_ = float(total_loss_)
    out_np = out.detach().cpu().numpy()[0]  #转化为np数组
    out_np_nonskull = out_np * img_np_mask
    # out_np_nonskull_gt = out_np_nonskull
    
    RMSE_out = np.sqrt((np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*31))
    MSE_out = (np.sum((out_np_nonskull-img_np_nonskull)**2))/(mask_num_npy*31)
    PSNR_out = 20*np.log10(1/RMSE_out)

    RMSE_noisy = np.sqrt((np.sum((out_np_nonskull-img_noisy_np_nonskull)**2))/(mask_num_npy*31))
    PSNR_noisy = 20*np.log10(1/RMSE_noisy)
    psrn_noisy_list.append(PSNR_noisy)
    psnr_noisy_array = np.array(psrn_noisy_list)

    psrn_out_list.append(PSNR_out)
    # mse_out_list.append(MSE_out)   
    rmse_out_list.append(RMSE_out)   
    total_loss_list.append(total_loss_)   #保存loss值 100epoch 一次  
    
    psnr_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/psnr_test.npy'
    psnr_noisy_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/psnr_noise_test.npy'
    # mse_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/mse_test.npy'
    rmse_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/rmse_test.npy'
    loss_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/psnr_rmse/test/loss_test.npy'
    psnr_array = np.array(psrn_out_list)
    np.save(psnr_name, np.array(psrn_out_list))
    np.save(psnr_noisy_name, np.array(psrn_noisy_list))
    np.save(rmse_name, np.array(rmse_out_list))   
    np.save(loss_name, np.array(total_loss_list))
    iteration_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/iterations/test/i.npy'  
    np.save(iteration_name, i)
    #保存每一次计算的PSNR

    # if (i+1)%show_every == 0:
    #     model_name_last = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/trained_model/test/epoch_' + str(i) +'.pt'
    #     torch.save(net.state_dict(),model_name_last)
    print ('Iteration %05d  Loss %f  PSNR_out: %f  PSNR_noisy_out: %f   RMSE_out: %f' % (i, total_loss.item(), PSNR_out, PSNR_noisy, RMSE_out), '\n', end='')
    if  i % show_every == 0:
        #每100个epoch 展示一下 去噪效果
        if i% (4*show_every) ==0:
            out_np = torch_to_np(out)
            X = 53
            
            out_np_X = out_np[0, :, :, X]
            out_np_Y = out_np[1, :, :, X]
            fig_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/result/test/epoch_' + str(i) +'.png'
            plt.figure(figsize=(8,1))
            plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1), plt.axis('off')
            plt.subplot(1,8,1), plt.imshow(img_np_nonskull[0, :, :, 53],vmin=0,vmax=1, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,2), plt.imshow(img_np_nonskull[1, :, :, 53],vmin=0,vmax=0.4, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,3), plt.imshow(img_noisy_np_nonskull[0, :, :, 53],vmin=0,vmax=1, cmap='gray'), plt.axis('off') 
            plt.subplot(1,8,4), plt.imshow(img_noisy_np_nonskull[1, :, :, 53],vmin=0,vmax=0.4, cmap='gray'), plt.axis('off')
            plt.subplot(1,8,5), plt.imshow(out_np_X,vmin=0,vmax=1,cmap='gray'), plt.axis('off')
            plt.subplot(1,8,6), plt.imshow(out_np_Y,vmin=0,vmax=1,cmap='gray') , plt.axis('off') 
            plt.subplot(1,8,7), plt.imshow((abs(img_np_nonskull[0, :, :, 53] - out_np_X)),vmin=0,vmax=0.3,cmap='gray') , plt.axis('off')
            plt.subplot(1,8,8), plt.imshow((abs(img_np_nonskull[1, :, :, 53] - out_np_Y)),vmin=0,vmax=0.12,cmap='gray') , plt.axis('off')
            plt.savefig(fig_name)
        loss_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/loss/test/epoch_' + str(i%(show_every*2)) +'.pt'
        model_name = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/trained_model/test/epoch_' + str(i) +'.pt'
        model_name_ = '/home/chenyunwei/MDIP_M1_W_3D/generate_data/model/test/epoch_' + str(i%(show_every*2)) +'.pt'
        torch.save(total_loss,loss_name)#保存模型
        torch.save(net.state_dict(),model_name_)
        torch.save(net.state_dict(),model_name)#保存模型

        
        if  i!=0 and i!=show_every and psnr_noisy_array[i-show_every+1:i+1].min() - psnr_noisy_array[i-2*show_every+1:i-show_every+1].min() < -0.05:  
            if last_i != i:
                t = 5
            if last_i == i:
                t = t-1
            last_i =i
            if t==0:
                if psnr_noisy_array[i].min() - psnr_noisy_array[i-show_every].min()<-0.5:
                    t=1
            if t >0:
                i = i-show_every
                # np.save(r'/home/hongzengcan/HZC/MDIP-M2-SD-3D/data/denoising/loss_psnr_array/Level10.01/iteration_Level10.01.npy', i)  
                net.load_state_dict(torch.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/model/test/epoch_' + str(i%(show_every*2)) +'.pt'))
                total_loss = torch.load('/home/chenyunwei/MDIP_M1_W_3D/generate_data/loss/test/epoch_' + str(i%(show_every*2)) +'.pt')
                total_loss.backward()
                for k in range(i+1,i+show_every+1):
                    del total_loss_list[i+1]
                    del psrn_out_list[i+1]
                    del rmse_out_list[i+1]
                    del psrn_noisy_list[i+1]
                total_loss_last = total_loss_list[i]
                out = net(net_input)
                out_inside_last = out.data
            else:
                total_loss_last = total_loss
        else:
            total_loss_last = total_loss
    LR=0.01*0.9**(i//2000)
    i += 1      
    return total_loss_list,psrn_out_list,rmse_out_list

p = get_params(OPT_OVER, net, net_input) 
optimizer = torch.optim.Adam(p, lr=LR)
for j in range(num_iter):
        optimizer.zero_grad()
        optimizer.step(closure)