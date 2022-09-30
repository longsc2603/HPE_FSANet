from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
# Local Imports
from transforms import Normalize, SequenceRandomTransform, ToTensor
from dataset import HeadposeDataset, DatasetFromSubset
from model import FSANet


train_transform = transforms.Compose([
            SequenceRandomTransform(),
            Normalize(mean=127.5, std=128),
            ToTensor()
            ])

validation_transform = transforms.Compose([Normalize(mean=127.5, std=128), ToTensor()])
data_path = '../data/type1/train'
hdb = HeadposeDataset(data_path, transform=None)

# Split dataset into train and validation
train_size = int(0.8 * len(hdb))
validation_size = len(hdb) - train_size
train_subset, validation_subset = random_split(hdb, [train_size, validation_size])

train_dataset = DatasetFromSubset(train_subset, train_transform)
validation_dataset = DatasetFromSubset(validation_subset, validation_transform)

del hdb, train_subset, validation_subset

print('Train Dataset Length: ', len(train_dataset))
print('Validation Dataset Length: ', len(validation_dataset))

# Setup dataloaders for train and validation
batch_size = 16

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

validation_loader = DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=False)

model = FSANet(var=False)

criterion = torch.nn.L1Loss()
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

device = torch.device("cuda")
model.to(device)


history = {'train_loss' : [], 'validation_loss' : []}
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
        best_loss = 100000
    
    # this is where we store all our losses during epoch before averaging them and storing in history
    iter_loss = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # prepare the model for training
        model.train()
        
        train_running_loss = []
        
        print(f'Epoch: {epoch}')
        # train on batches of data, assumes you already have train_loader
        print("Printing Train Loss...")
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding poses
            images,gt_poses = data
            
            # put data inside gpu
            images = images.float().to(device)
            gt_poses = gt_poses.float().to(device)

            # call model forward pass
            predicted_poses = model(images)

            # calculate the softmax loss between predicted poses and ground truth poses
            loss = criterion(predicted_poses, gt_poses)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()
            
            # get loss
            loss_scalar = loss.item()

            #update loss logs
            train_running_loss.append(loss_scalar)
            
            #collect batch losses to compute epoch loss later
            iter_loss.append(loss_scalar)
            
            if batch_i % 1000 == 999:    # print every 100 batches
                print(f'Batch: {batch_i+1}, Avg. Train Loss: {np.mean(train_running_loss)}')
                train_running_loss = []
        else: 
            history['train_loss'].append(np.mean(iter_loss))
            iter_loss.clear()
            validation_running_loss = []
            model.eval()
            with torch.no_grad():
                print("Printing Validation Loss...")
                for batch_i, data in enumerate(validation_loader):
                    # get the input images and their corresponding poses
                    images,gt_poses = data

                    # put data inside gpu
                    images = images.float().to(device)
                    gt_poses = gt_poses.float().to(device)

                    # call model forward pass
                    predicted_poses = model(images)

                    # calculate the softmax loss between predicted poses and ground truth poses
                    loss = criterion(predicted_poses, gt_poses)
                    
                    #convert loss into a scalar using .item()
                    loss_scalar = loss.item()
                    
                    #add loss to the running_loss, use
                    validation_running_loss.append(loss_scalar)
                    
                    #collect batch losses to compute epoch loss later
                    iter_loss.append(loss_scalar)
                    
                    
                    if batch_i % 1000 == 999:    # print every 10 batches
                        print(f'Batch: {batch_i+1}, Avg. Validation Loss: {np.mean(validation_running_loss)}')
                        validation_running_loss = []
                        
            history['validation_loss'].append(np.mean(iter_loss))
            iter_loss.clear()
            
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
                            }, '../checkpoints/8_8_2020_temp2.chkpt')
                
                print(f'Model improved since last epoch! New Best Val Loss: {best_loss}')
                
        # update lr schedular
        scheduler.step()                                     

    print('Finished Training')


try:
    # train your network
    n_epochs = 90 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs)
except KeyboardInterrupt:
    print('Stopping Training...')