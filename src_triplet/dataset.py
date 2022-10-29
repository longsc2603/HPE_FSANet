"""
Dataset Class for FSANet Training
Implemented by Omar Hassan
August, 2020
"""
"""
Modified by Tran Van Long for training FSANet
based on Triplet Network architecture
October, 2022
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random


random.seed(a=28102022)
np.random.seed(28102022)


class Cutout():
    def __init__(self, length, holes=1):
        self._length = length
        self._holes = holes

    def __call__(self, image):
        for _ in range(self._holes):
            h, w, _ = image.shape
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(y - self._length, 0)
            y2 = min(y + self._length, h)
            x1 = max(x - self._length, 0)
            x2 = min(x + self._length, w)
            image[y1:y2, x1:x2] = 0

        return image


# this function is used to transform dataset to triplets for triplet network
def triplet_transform(x, y, binned_labels):
    yaw = dict()
    pitch = dict()
    roll = dict()
    triplet_x = []
    triplet_y = []
    # create 3 dicts holding indexes for each corresponding bin value
    for index in range(len(binned_labels[0])):
        if binned_labels[0, index].item() not in yaw.keys():
            yaw[binned_labels[0, index].item()] = []
            yaw[binned_labels[0, index].item()].append(index)
        else:
            yaw[binned_labels[0, index].item()].append(index)
        if binned_labels[1, index].item() not in pitch.keys():
            pitch[binned_labels[1, index].item()] = []
            pitch[binned_labels[1, index].item()].append(index)
        else:
            pitch[binned_labels[1, index].item()].append(index)
        if binned_labels[2, index].item() not in roll.keys():
            roll[binned_labels[2, index].item()] = []
            roll[binned_labels[2, index].item()].append(index)
        else:
            roll[binned_labels[2, index].item()].append(index)
    for index in range(x.shape[0]):
        similar = -1        # holds index of sample similar to the anchor
        different = -1      # holds index of sample different than the anchor
        # most_similar = -1    # holds index of sample most similar to the anchor
        all_similar = []
        # all_most_similar = []
        for id in yaw[binned_labels[0, index].item()]:
            if False not in np.array(binned_labels[:, index] == binned_labels[:, id]):
                all_similar.append(id)
        if len(all_similar) == 0:
            continue
        similar = random.choice(all_similar)
        while different == -1:
            temp = random.randrange(x.shape[0])
            if temp == index or temp == similar:
                continue
            if False in np.array(binned_labels[:, index] == binned_labels[:, temp]):
                different = temp
        if similar != -1:
            # if similar and different samples are both found, append them
            triplet_x.append([x[index, :], x[similar, :], x[different, :]])
            triplet_y.append([y[index, :], y[similar, :], y[different, :]])

    return triplet_x, triplet_y


class HeadposeDataset(Dataset):
    def __init__(self, data_path):
        image = []
        pose = []
        data = np.load(f'{data_path}.npz')
        image.append(data["image"])
        pose.append(data["pose"])

        image = np.concatenate(image, 0)
        pose = np.concatenate(pose, 0)

        # exclude examples with pose outside [-99,99]
        x_data = []
        y_data = []
        for i in range(pose.shape[0]):
            if np.max(pose[i, :]) <= 99.0 and np.min(pose[i, :]) >= -99.0:
                x_data.append(image[i])
                y_data.append(pose[i])

        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        print('x (images) shape: ', self.x_data.shape)
        print('y (poses) shape: ', self.y_data.shape)


    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y


# used to apply different transforms to train, validation dataset
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None, triplet=False):
        self.subset = subset
        self.transform = transform
        self.triplet_transform = triplet
        self.x_data = []
        self.y_data = []
        for id in range(len(self.subset)):
            x, y = self.subset[id]
            self.x_data.append(x)
            self.y_data.append(y)
        self.x_data = np.array(self.x_data, dtype='float32')
        self.y_data = np.array(self.y_data, dtype='float32')
        if self.transform:
            for id in range(self.x_data.shape[0]):
                self.x_data[id] = self.transform(self.x_data[id])
        if self.triplet_transform:
            # Cutout data augmentation, from x training images to 2x images
            cutout = Cutout(length=8)
            for id in range(self.x_data.shape[0]):
                cutout_img = np.expand_dims(cutout(self.x_data[id]), axis=0)
                pose = np.expand_dims(self.y_data[id], axis=0)
                self.x_data = np.append(self.x_data, cutout_img, axis=0)
                self.y_data = np.append(self.y_data, pose, axis=0)
            print('The number of triplets is doubled due to cutout augmentation')

            yaw = self.y_data[:, 0]
            pitch = self.y_data[:, 1]
            roll = self.y_data[:, 2]

            # Bin values
            bins = np.array(range(-99, 102, 3))
            binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
            binned_labels = torch.LongTensor(binned_pose)

            self.triplet_x, self.triplet_y = triplet_transform(self.x_data, self.y_data, binned_labels)
            self.triplet_x = np.array(self.triplet_x)
            self.triplet_y = np.array(self.triplet_y)
            print('Training images (triplet) shape: ', self.triplet_x.shape)
            print('Training poses (triplet) shape: ', self.triplet_y.shape)
        else:
            print('Validation images shape: ', self.x_data.shape)
            print('Validation poses shape: ', self.y_data.shape)

    def __getitem__(self, index):
        if self.triplet_transform:
            x = self.triplet_x[index]
            y = self.triplet_y[index]
        else:
            x = self.x_data[index]
            y = self.y_data[index]

        return x, y

    def __len__(self):
        if self.triplet_transform:
            return self.triplet_x.shape[0]
        else:
            return self.y_data.shape[0]
