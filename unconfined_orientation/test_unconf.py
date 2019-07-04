import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import figure
import matplotlib.gridspec as gridspec
from ansim_dataset_unconf import ansimDataset, create_circular_mask, ansimDataset_orientation
from ConvLSTM_unconf import MtConvLSTM
import random
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import scipy.ndimage
import cv2
import scipy.misc
from array2gif import write_gif
from scipy.misc import imsave

#%env CUDA_VISIBLE_DEVICES=1

img_path = '/home/rliu/ansim/data/unconfined_steph/cropped_orientation/'
img_list_csv = '/home/rliu/github/ansim/unconfined_orientation/img_list.csv'
train_csv = '/home/rliu/github/ansim/unconfined_orientation/train_unconf.csv'
test_csv = '/home/rliu/github/ansim/unconfined_orientation/test_unconf.csv'

mask = create_circular_mask(128,128)

mask = create_circular_mask(128,128)
orientation_set = ansimDataset_orientation(img_list_csv = img_list_csv, seq_csv = test_csv, root_dir = img_path, step=23, random_rotate = False, transform=None, image_size = 128, rand_range=0)
testloader = torch.utils.data.DataLoader(orientation_set, batch_size=1, shuffle=False,
                                                     num_workers=1)

model = torch.load('/home/rliu/ansim/models/dataset3/6-26_mt_paper_steph_predict/0160.weights').cuda()
try:
    os.mkdir('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20')
except OSError as exc:
    pass


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU in use")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    i = 0
    for data in testloader:
        i += 1
        try:
            os.mkdir('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d' % i)
            os.mkdir('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/inputs' % i)
            os.mkdir('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/target' % i)
            os.mkdir('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/predicted' % i)
        except OSError as exc:
            pass

        data_split = torch.split(data, (8,15), dim=1)
        inputs = data_split[0]
        target = data_split[1]

        if use_gpu:
            inputs, target = inputs.to(device), target.to(device)
        else:
            inputs, target = Variable(inputs), Variable(target)

        layer_output_list, last_state_list, pred_output, pred_a = model(inputs)
        
#         a = (layer_output_list, last_state_list, pred_output, pred_a)
# #         with open('data.pickle', 'wb') as f:
# #             pickle.dump(a, f)
# #         with open('data.pickle', 'rb') as f:
# #             a = pickle.load(f)
#         (layer_output_list, last_state_list, pred_output, pred_a) = a 
        _,_,_, pred_b = model.module.forecast(layer_output_list, last_state_list, pred_output, pred_a, predict_steps = 7)
        
        predicted = torch.cat((pred_a, pred_b), 1)
        
        predicted_1 = predicted[0,:,0,:,:].cpu().detach().numpy()
        target_1 = target[0,:,0,:,:].cpu().detach().numpy()
        inputs_1 = inputs[0,:,0,:,:].cpu().detach().numpy()
        
        predicted_2 = predicted[0,:,1,:,:].cpu().detach().numpy()
        target_2 = target[0,:,1,:,:].cpu().detach().numpy()
        inputs_2 = inputs[0,:,1,:,:].cpu().detach().numpy()
        
        
        
        for ii in range(inputs_1.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/inputs/inputs_1_%0.4d.png' % (i, ii), inputs_1[ii,:,:])
        for ii in range(target_1.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/target/target_1_%0.4d.png' % (i, ii), target_1[ii,:,:])
        for ii in range(predicted_1.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/predicted/predicted_1_%0.4d.png' % (i, ii), predicted_1[ii,:,:])
            
        for ii in range(inputs_2.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/inputs/inputs_2_%0.4d.png' % (i, ii), inputs_2[ii,:,:])
        for ii in range(target_2.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/target/target_2_%0.4d.png' % (i, ii), target_2[ii,:,:])
        for ii in range(predicted_2.shape[0]):
            imsave('/home/rliu/ansim/results/test_6-27_orientation_160weights_predict20/%0.4d/predicted/predicted_2_%0.4d.png' % (i, ii), predicted_2[ii,:,:])
            

# inputs = []
# for i in range(257):
#     for j in range(10):
#         filename = '/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/inputs/inputs%0.4d.png' % (i+1,j)
#         inputs.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/inputs.gif' % (i+1), inputs, format='GIF', fps=3)

# target = []
# for i in range(257):
#     for j in range(30):
#         filename = '/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/target/target%0.4d.png' % (i+1,j)
#         target.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/target.gif' % (i+1), target, format='GIF', fps=3)

# predicted = []
# for i in range(257):
#     for j in range(30):
#         filename = '/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/predicted/predicted%0.4d.png' % (i+1,j)
#         predicted.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_6-19_predict 20/%0.4d/predicted.gif' % (i+1), predicted, format='GIF', fps=3)
            
    
    
    
    