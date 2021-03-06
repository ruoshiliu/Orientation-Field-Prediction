import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ansim_dataset200 import ansimDataset, create_circular_mask
# from convolution_lstm import encoderConvLSTM, decoderConvLSTM
from ConvLSTM200 import MtConvLSTM
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
from scipy.misc import imsave
import imageio

img_path = '/home/rliu/ansim/data/data/JPEGImages/'
img_list_csv = '/home/rliu/github/ansim/img_list.csv'
train_csv = '/home/rliu/github/ansim/d200um/train200.csv'
test_csv = '/home/rliu/github/ansim/d200um/test200.csv'
output_path = '/home/rliu/ansim/models/dataset2/4-25_mt-6-8-10-15/0240.weights'

mask = create_circular_mask(128,128)

testset = ansimDataset(img_list_csv = img_list_csv, seq_csv = test_csv, root_dir = img_path, step=40, random_rotate = False, transform=None, image_size = 128, rand_range=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                                     num_workers=1)

model = torch.load('/home/rliu/ansim/models/dataset2/4-25_mt-6-8-10-12/0235.weights').cuda()
try:
    os.mkdir('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40')
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
            os.mkdir('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d' % i)
            os.mkdir('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/inputs' % i)
            os.mkdir('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/target' % i)
            os.mkdir('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/predicted' % i)
        except OSError as exc:
            pass

        data_split = torch.split(data, (10,30), dim=1)
        inputs = data_split[0]
        target = data_split[1]

        if use_gpu:
            inputs, target = inputs.to(device), target.to(device)
        else:
            inputs, target = Variable(inputs), Variable(target)

        layer_output_list, last_state_list, pred_output, pred_a = model(inputs)
        
        a = (layer_output_list, last_state_list, pred_output, pred_a)
#         with open('data.pickle', 'wb') as f:
#             pickle.dump(a, f)
#         with open('data.pickle', 'rb') as f:
#             a = pickle.load(f)
        (layer_output_list, last_state_list, pred_output, pred_a) = a 
        _,_,_, pred_b = model.module.forecast(layer_output_list, last_state_list, pred_output, pred_a, predict_steps = 20)
        
        predicted = torch.cat((pred_a, pred_b), 1)
        
        predicted = predicted[0,:,0,:,:].cpu().detach().numpy()
        target = target[0,:,0,:,:].cpu().detach().numpy()
        inputs = inputs[0,:,0,:,:].cpu().detach().numpy()
        
        for ii in range(inputs.shape[0]):
            imsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/inputs/inputs%0.4d.png' % (i, ii), inputs[ii,:,:])
        for ii in range(target.shape[0]):
            imsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/target/target%0.4d.png' % (i, ii), target[ii,:,:])
        for ii in range(predicted.shape[0]):
            imsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/predicted/predicted%0.4d.png' % (i, ii), predicted[ii,:,:])
            

# inputs = []
# for i in range(257):
#     for j in range(10):
#         filename = '/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/inputs/inputs%0.4d.png' % (i+1,j)
#         inputs.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/inputs.gif' % (i+1), inputs, format='GIF', fps=3)

# target = []
# for i in range(257):
#     for j in range(30):
#         filename = '/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/target/target%0.4d.png' % (i+1,j)
#         target.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/target.gif' % (i+1), target, format='GIF', fps=3)

# predicted = []
# for i in range(257):
#     for j in range(30):
#         filename = '/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/predicted/predicted%0.4d.png' % (i+1,j)
#         predicted.append(imageio.imread(filename))
#     imageio.mimsave('/home/rliu/ansim/results/test_4-25_mt-6-8-10-12_predict40/%0.4d/predicted.gif' % (i+1), predicted, format='GIF', fps=3)
            
    
    
    
    