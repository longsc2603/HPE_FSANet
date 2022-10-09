import numpy as np



np.random.seed(42)
file = np.load(".\\BIWI_train1.npz")
out_imgs = []
out_poses = []
su = 0
img_size =64

#loading data from file.npz

for i in range(0,4):
    file = np.load(f".\\BIWI_train{i+1}.npz")
    su+=file['image'].shape[0]
    for value in file['image']:
        out_imgs.append(value)
    for value in file['pose']:
        out_poses.append(value)
#print(f"sum: {su}")
# out_imgs=np.array(out_imgs)
# out_poses=np.array(out_poses)
randFlag = np.zeros(su)
randFlag[0:int(su*0.7)] = 1
randFlag = np.random.permutation(randFlag)
out_imgs_training=[]
out_imgs_test = []



# 70/30 images data
for idx,value in enumerate(out_imgs):
    if randFlag[idx] == 1:
        out_imgs_training.append(value)
    else:
        out_imgs_test.append(value)





# 70/30 pose data
out_pose_training=[]
out_pose_test = []
for idx,value in enumerate(out_poses):
    if randFlag[idx] == 1:
        out_pose_training.append(value)
    else:
        out_pose_test.append(value)




# pack:
np.savez('.\\BIWI_train.npz',image=np.array(out_imgs_training), pose=np.array(out_pose_training), img_size=img_size)
np.savez('.\\BIWI_test.npz',image=np.array(out_imgs_test), pose=np.array(out_pose_test), img_size=img_size)