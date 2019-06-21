import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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

img_path = '/work/ruoshiliu/ansim/data/unconfined_steph/cropped_orientation/'
img_list_csv = '/home/ruoshiliu/github/ansim/unconfined_orientation/img_list.csv'
train_csv = '/home/ruoshiliu/github/ansim/unconfined_orientation/train_unconf.csv'
test_csv = '/home/ruoshiliu/github/ansim/unconfined_orientation/test_unconf.csv'
output_path = '/home/ruoshiliu/ansim/models/dataset3/6-17_mt_paper_orientation/final.weights'

mask = create_circular_mask(128,128)
trainset = ansimDataset_orientation(img_list_csv = img_list_csv, seq_csv = train_csv, root_dir = img_path, step=10, random_rotate = True, transform=None)
trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=8, shuffle=True,
                                             num_workers=2)

testset = ansimDataset_orientation(img_list_csv = img_list_csv, seq_csv = test_csv, root_dir = img_path, step=10, random_rotate = False, transform=None)
testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=8, shuffle=False,
                                             num_workers=2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU in use")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_workers = 2,  num_epochs=25, batch_size = 4, step_size = 20, image_size = 128):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    epoch_num = 0
    for epoch in range(num_epochs):
        epoch_num += 1
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        

        # Iterate over data.
        trainset = ansimDataset_orientation(img_list_csv = img_list_csv, seq_csv = train_csv, root_dir = img_path, step=step_size, random_rotate = True, transform=None, image_size = image_size, rand_range=10)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                     batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers)

        print("trainloader ready!")
        testset = ansimDataset_orientation(img_list_csv = img_list_csv, seq_csv = test_csv, root_dir = img_path, step=step_size, random_rotate = False, transform=None, image_size = image_size, rand_range=0)
        testloader = torch.utils.data.DataLoader(testset,
                                                     batch_size=10, shuffle=False,
                                                     num_workers=num_workers)
        print("testloader ready!")
        
        for data in trainloader:
            # get the inputs
            data_split = torch.split(data, int(data.shape[1]/2), dim=1)
            inputs = data_split[0]
            target = data_split[1]

            # wrap them in Variable
            if use_gpu:
                inputs, target = inputs.to(device), target.to(device)
            else:
                inputs, target = Variable(inputs), Variable(target)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            _, _, _, predicted = model(inputs)

            loss = criterion(predicted, target)

            loss.backward()
            optimizer.step()

            # statistics
            loss = loss*1000
            iter_loss = loss.item()
            running_loss += loss.item()    
            epoch_loss = running_loss / len(trainset)
            
            print('{} Loss: {:.4f} batch_loss: {:f}'.format(
                "train", epoch_loss, iter_loss))
        
        with torch.no_grad():
            running_loss_test = 0.0
            loss_by_class = 0.0
            test_iter = 0
            for data in testloader:
                test_iter += 1
                data_split = torch.split(data, int(data.shape[1]/2), dim=1)
                inputs = data_split[0]
                target = data_split[1]
                
                
                if use_gpu:
                    inputs, target = inputs.to(device), target.to(device)
                else:
                    inputs, target = Variable(inputs), Variable(target)


                _, _, _, predicted = model(inputs)

                
                loss_test = criterion(predicted, target)
                loss_test = loss_test * 1000
                iter_loss_test = loss_test.item()
                running_loss_test += loss_test.item()   
                epoch_loss_test = running_loss_test / len(testset)
                loss_by_class += loss_test.item()
                if test_iter == 20:
                    print('Loss on the 1-200: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 40:
                    print('Loss on the 201-400: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 50:
                    print('Loss on the 401-500: %.5f ' % (loss_by_class/100.0))
                    loss_by_class = 0.0
                elif test_iter == 70:
                    print('Loss on the 501-700: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 80:
                    print('Loss on the 701-800: %.5f ' % (loss_by_class/100.0))
                    loss_by_class = 0.0
                epoch_loss_test = running_loss_test / len(testset)
            
            print('Loss on the test images: %.5f ' % (epoch_loss_test))
        
        if epoch_num % 10 == 0 or epoch_num == 1:
            print('saving wiehgts...')
            output_path = "/home/ruoshiliu/ansim/models/dataset3/6-17_mt_paper_orientation/%0.4d.weights" % (epoch_num)
            torch.save(model, output_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
#	print('saving wiehgts.../n')
#	output_path = sprintf("/home/ruoshiliu/ansim/models/%0.4d.weights" % epoc
    return model
# transfer learning resnet18
step_size = 10

model = MtConvLSTM(input_size=(128,128),
                 input_dim=2,
                 hidden_dim=[[16,32,64],[16,32,64],[32,64,128],[32,64,128,128]],
                 kernel_size=[[3,3,3],[5,3,3],[5,5,5],[7,5,5,5]],
                 num_layers=[3,3,3,4],
                 predict_steps=5,
                 batch_first=True,
                 num_scale=4,
                 bias=True,
                 return_all_layers=True,
                 interpolation = 0)
print(model)
# model = ConvLSTM(input_size=(128,128),
#                  input_dim=1,
#                  hidden_dim=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32],
#                  kernel_size=[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
#                  num_layers=20,
#                  predict_steps=int(step_size/2),
#                  batch_first=True,
#                  bias=True,
#                  return_all_layers=True)


count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model parameter: %d" % count_param)

    

if use_gpu:
    model = torch.nn.DataParallel(model)
    model.to(device)

criterion = nn.MSELoss()


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.5)

# train model
model = train_model(model, criterion, optimizer_ft, 
            exp_lr_scheduler,
            batch_size = 16,
            step_size = 10,
            num_epochs = 240,
            num_workers = 1,
            image_size = 128)
torch.save(model, output_path)
