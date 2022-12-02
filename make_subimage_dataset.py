import os
import cv2
import tqdm
import numpy as np
import pickle as pkl

img_list = os.listdir("datasets/T91")

subimages_hr = []
subimages_lr = []

for filename in tqdm.tqdm(img_list):
    if filename.endswith(".png"):
        img = cv2.imread(f"datasets/T91/{filename}")#.astype(float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("img", img)
        # cv2.waitKey(100)
        img_lr = cv2.resize(img, (img.shape[1]//4,img.shape[0]//4),interpolation=cv2.INTER_CUBIC)
        img_lr = cv2.resize(img_lr, (img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
        img = img.astype(float)/255.
        img_lr = img_lr.astype(float)/255.
        for i in range(0, img.shape[0]-31, 4):
            for j in range(0, img.shape[1]-31, 4):
                subimages_hr.append(img[i:i+32,j:j+32])
                subimages_lr.append(img_lr[i:i+32,j:j+32])
# print(np.array(subimages).shape)
subimages_hr = np.array(subimages_hr)
subimages_lr = np.array(subimages_lr)
with open("dataset_hr.pkl","wb") as f:
    pkl.dump(subimages_hr, f)
with open("dataset_lr.pkl","wb") as f:
    pkl.dump(subimages_lr, f)
