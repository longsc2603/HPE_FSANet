import numpy as np
# import os
# from torch.utils.data import Dataset
# from tqdm import tqdm


np.random.seed(28102022)

"""
class Dataset(Dataset):
    def __init__(self, path, transform=None):
        data = np.load(path)
        self.image = data['image']
        self.pose = data['pose']
        self.samples = data['image'].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        image = self.image[index]
        pose = self.pose[index]
        if self.transform:
            image = self.transform(image)
        return image, pose

    def __len__(self):
        return self.samples
"""


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


"""
def main():
    for f in os.listdir(".//data"):
        if f.endswith(".npz"):
            path = f".//data//{f}"
            fn, ftext = os.path.splitext(f)
            dataset1 = Dataset(path, transform=Cutout(8))
            dataset2 = Dataset(path)
            # for i in range(1):
            #     image,pose = dataset[i]
            #     cv2.imshow('image',image)
            #     cv2.waitKey(0)
            out_imgs = []
            out_pose = []

            for i in tqdm(range(len(dataset1)), desc=f'{fn} dataset'):
                image1, pose1 = dataset1[i]
                out_imgs.append(image1)
                out_pose.append(pose1)
                image2, pose2 = dataset2[i]
                out_imgs.append(image2)
                out_pose.append(pose2)
                # cv2.imshow("image",image1)
                # cv2.waitKey(50)
            np.savez(f".//cutoutdata//{fn}.npz", image=
                     np.array(out_imgs), pose=np.array(out_pose), img_size=64)

    return


if __name__ == "__main__":
    main()
"""
