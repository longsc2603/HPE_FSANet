"""
Modified by Tran Van Long for training FSANet
based on Triplet Network architecture
October, 2022
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random
from sklearn.model_selection import train_test_split
# Local Imports
from dataset import HeadposeDataset, DatasetFromSubset
from model import FSANet, TotalLoss
from transforms import SequenceRandomTransform, Normalize


np.random.seed(28102022)
random.seed(a=28102022)
data_path = '/home/guest/long.tran/headpose-fsanet-pytorch/data/BIWI_train'
hdb = HeadposeDataset(data_path)
train_transform = transforms.Compose([
            SequenceRandomTransform(),
            Normalize(mean=127.5, std=128),
            ])

val_transform = transforms.Compose([Normalize(mean=127.5, std=128)])
# Split dataset into train and validation
train_subset, validation_subset = train_test_split(hdb, test_size=0.2, random_state=28102022)  # ddmmyyyy
del hdb

train_dataset = DatasetFromSubset(train_subset, train_transform, triplet=True)
validation_dataset = DatasetFromSubset(validation_subset, val_transform)

# Setup dataloaders for train and validation
batch_size = 32

# for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(28102022)
torch.cuda.manual_seed(28102022)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

validation_loader = DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=True)

model = FSANet(var=False)
train_loss_f = TotalLoss(margin=0.2)
eval_loss_f = torch.nn.L1Loss()
learning_rate = 75e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

device = torch.device("cpu")
print(f"Using {device}")
model = model.to(device)

history = {'train_loss': [], 'validation_loss': []}
# This holds best states wrt validation loss that we are saving,
# we can use these to resume from last best loss or keep the best one for inference later. 
best_states = {}

# This is where we save our best state_dict during training and validation
best_states['model'] = model.state_dict()

# This is for storing optim state if we wish to continue training from our last best state
best_states['optim'] = optimizer.state_dict()


def train_net(n_epochs):
    global best_states, history

    # average test loss over epoch used to find best model parameters
    if(len(history['validation_loss']) > 0):
        min_loss_idx = np.argmin(history['validation_loss'])
        best_loss = history['validation_loss'][min_loss_idx]
    else:
        best_loss = 10000
    
    # this is where we store all our losses during epoch before averaging them and storing in history
    itter_loss = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # prepare the model for training
        model.train()
        
        train_running_loss = []
        
        print(f'Epoch: {epoch+1}')
        # train on batches of data, assumes you already have train_loader
        print("Printing Train Loss...")
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding poses
            images, gt_poses = data

            # put data inside gpu
            images = torch.moveaxis(images, (2, 3), (3, 4)).float().to(device)
            # The above line is to change tensor with size
            # [batch_size, 3, 64, 64, 3] into [batch_size, 3, 3, 64, 64]
            # as needed to feed to the model.

            gt_poses = gt_poses.float().to(device)

            # call model forward pass
            predicted_anchor, anchor_features = model(images[:, 0], num_image=3)
            predicted_positive, positive_features = model(images[:, 1], num_image=3)
            predicted_negative, negative_features = model(images[:, 2], num_image=3)

            anchor_features = torch.stack(anchor_features)
            positive_features = torch.stack(positive_features)
            negative_features = torch.stack(negative_features)
            # The features output are in the form: list(3xTensor(16)),
            # which is a 16x1 Tensor for 3 stages. So I use torch.stack
            # to get a tensor with size 16x3

            # calculate the triplet loss
            loss = train_loss_f(anchor_features, positive_features, negative_features,
                                predicted_anchor, predicted_positive, predicted_negative,
                                gt_poses)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # get loss
            loss_scalar = loss.item()

            # update loss logs
            train_running_loss.append(loss_scalar)

            # collect batch losses to compute epoch loss later
            itter_loss.append(loss_scalar)

            # print every 100 batches
            if batch_i % 100 == 99:
                print(f'Batch: {batch_i+1}, Avg. Train Loss: {np.mean(train_running_loss):.5f}')
                train_running_loss = []
        else:
            history['train_loss'].append(np.mean(itter_loss))
            itter_loss.clear()
            validation_running_loss = []
            model.eval()
            with torch.no_grad():
                print("Printing Validation Loss...")
                for batch_i, data in enumerate(validation_loader):
                    # get the input images and their corresponding poses
                    images, gt_poses = data

                    # put data inside gpu
                    images = torch.moveaxis(images, (1, 2), (2, 3)).float().to(device)
                    # The above line is to change tensor with size
                    # [batch_size, 64, 64, 3] into [batch_size, 3, 64, 64]
                    # as needed to feed to the model.

                    gt_poses = gt_poses.float().to(device)

                    # call model forward pass
                    predicted_poses = model(images)

                    # calculate the softmax loss between predicted poses and ground truth poses
                    loss = eval_loss_f(predicted_poses, gt_poses)
                    
                    # convert loss into a scalar using .item()
                    loss_scalar = loss.item()
                    
                    # add loss to the running_loss, use
                    validation_running_loss.append(loss_scalar)
                    
                    # collect batch losses to compute epoch loss later
                    itter_loss.append(loss_scalar)
                    
                    # print every 100 batches
                    if batch_i % 100 == 99:
                        print(f'Batch: {batch_i+1}, Avg. Validation Loss: {np.mean(validation_running_loss):.5f}')
                        validation_running_loss = []
                        
            history['validation_loss'].append(np.mean(itter_loss))
            itter_loss.clear()
            
            #if current is better than previous, update state_dict and store current as best
            if(history['validation_loss'][-1] < best_loss):
                best_loss = history['validation_loss'][-1]
                best_states['model'] = model.state_dict()
                best_states['optim'] = optimizer.state_dict()
                #Save Model Checkpoint
                torch.save({
                            'total_epochs': len(history['train_loss']),
                            'best_states': best_states,
                            'history' : history
                            }, 'checkpoints/27_10_2022.chkpt')
                
                print(f'Model improved since last epoch! New Best Validation Loss: {best_loss:.5f}')
                
        # update lr schedular
        scheduler.step()                                     

    print('Finished Training')


try:
    # train your network
    n_epochs = 100 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs)
except KeyboardInterrupt:
    print('Stopping Training...')
