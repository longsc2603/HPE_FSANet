from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import time
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
#Local Imports
from transforms import Normalize,ToTensor
from dataset import HeadposeDataset
from model import FSANet


transform = transforms.Compose([
            Normalize(mean=127.5,std=128),
            ToTensor()
            ])

data_path = '/home/guest/long.tran/headpose-fsanet-pytorch/data/BIWI_test'
hdb = HeadposeDataset(data_path, transform=transform)

#Setup dataloaders for train and validation
batch_size = 64

test_loader = DataLoader(hdb, 
                          batch_size=batch_size,
                          shuffle=False)

model = FSANet(var=False)
#Load Model Checkpoint
chkpt_dic = torch.load('../checkpoints/27_09_2022.chkpt')
model.load_state_dict(chkpt_dic['best_states']['model'])

#Place all the necessary things in GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model.to(device)


def test_net():
    yaw_loss = []
    pitch_loss = []
    roll_loss = []
    model.eval()
    with torch.no_grad():
        for batch_i, data in enumerate(test_loader):
            # get the input images and their corresponding poses
            images,gt_poses = data

            # put data inside gpu
            images = images.float().to(device)
            gt_poses = gt_poses.float().to(device)

            # call model forward pass
            predicted_poses = model(images)

            abs_loss = torch.abs(gt_poses-predicted_poses)
            
            abs_loss = abs_loss.cpu().numpy().mean(axis=0)
            
            yaw_loss.append(abs_loss[0])
            pitch_loss.append(abs_loss[1])
            roll_loss.append(abs_loss[2])
    
    yaw_loss = np.mean(yaw_loss)
    pitch_loss = np.mean(pitch_loss)   
    roll_loss = np.mean(roll_loss)
    print('Mean Absolute Error:')
    print(f'Yaw: {yaw_loss:.2f}, Pitch: {pitch_loss:.2f}, Roll: {roll_loss:.2f}')


try:
    # test your network
    test_net()
except KeyboardInterrupt:
    print('Stop Testing...')