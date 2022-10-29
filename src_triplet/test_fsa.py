# from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
# Local Imports
# from transforms import Normalize
from dataset import HeadposeDataset
from model import FSANet

"""
transform = transforms.Compose([
            Normalize(mean=127.5,std=128),
            ])
"""
data_path = '/home/guest/long.tran/headpose-fsanet-pytorch/data/BIWI_test'
hdb = HeadposeDataset(data_path)

#Setup dataloaders for train and validation
batch_size = 32

torch.backends.cudnn.deterministic = True   # for reproducibility
torch.manual_seed(28102022)
torch.cuda.manual_seed(28102022)

test_loader = DataLoader(hdb,
                         batch_size=batch_size,
                         shuffle=True)

model = FSANet(var=False)
# Load Model Checkpoint
chkpt_dic = torch.load('checkpoints/27_10_2022.chkpt')
model.load_state_dict(chkpt_dic['best_states']['model'])

# Place all the necessary things in GPU
device = torch.device("cpu")
print(f"Using {device}")
model.to(device)


def test_net():
    yaw_loss = []
    pitch_loss = []
    roll_loss = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # get the input images and their corresponding poses
            images, gt_poses = data

            # put data inside device
            images = torch.moveaxis(images, (1, 2), (2, 3)).float().to(device)
            # The above line is to change tensor with size
            # [batch_size, 64, 64, 3] into [batch_size, 3, 64, 64]
            # as needed to feed to the model.
            # That line and the normalizing line below are supposed to be in
            # a transformation step when dataset is created, but for many
            # reasons and bugs, I choose to do it like this.
            
            gt_poses = gt_poses.float().to(device)

            # call model forward pass
            # normalizing images with mean 127.5 and std 128
            predicted_poses = model(torch.div(images - 127.5, 128))

            abs_loss = torch.abs(gt_poses-predicted_poses)
            abs_loss = abs_loss.cpu().numpy().mean(axis=0)

            yaw_loss.append(abs_loss[0])
            pitch_loss.append(abs_loss[1])
            roll_loss.append(abs_loss[2])
    
    yaw_loss = np.mean(yaw_loss)
    pitch_loss = np.mean(pitch_loss)   
    roll_loss = np.mean(roll_loss)
    print('Mean Absolute Error:')
    print(f'Yaw: {yaw_loss:.3f}, Pitch: {pitch_loss:.3f}, Roll: {roll_loss:.3f}')
    print(f'Average: {(yaw_loss + pitch_loss + roll_loss)/3:.3f}')


try:
    # test your network
    test_net()
except KeyboardInterrupt:
    print('Stop Testing...')
