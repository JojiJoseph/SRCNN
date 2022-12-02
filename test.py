from model import SRCNN
import torch
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
if torch.cuda.is_available():
    device  = torch.device("cuda")
else:
    device = torch.device("cpu")

model = SRCNN().to(device)

model.load_state_dict(torch.load("model.th",map_location=device))

for filename in os.listdir("datasets/Set14"):
    if not filename.endswith(".png"):
        continue
    img1 = cv2.imread(f"datasets/Set14/{filename}")
    img1_lr = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2),interpolation=cv2.INTER_CUBIC)
    img1_lr = cv2.resize(img1_lr, (img1.shape[1], img1.shape[0]),interpolation=cv2.INTER_CUBIC)

    img_in = cv2.cvtColor(img1_lr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img_in_tensor = torch.from_numpy(img_in).to(device).permute(2,0,1)
    y = model(img_in_tensor[None,:,:,:])
    y = torch.clamp(y,0.,1.)*255
    y = y[0].permute(1,2,0).cpu().detach().numpy().astype(np.uint8)

    print(f"\n{filename}")
    print("ssim between gt and y", structural_similarity(img1[:,:,::-1], y, channel_axis=2))
    print("ssim between gt and lr", structural_similarity(img1[:,:,::-1], img1_lr[:,:,::-1], channel_axis=2))
    print("psnr between gt and y", peak_signal_noise_ratio(img1[:,:,::-1], y))
    print("psnr between gt and lr", peak_signal_noise_ratio(img1[:,:,::-1], img1_lr[:,:,::-1]))
    cv2.imshow("gt", img1)
    cv2.imshow("lr", img1_lr)
    cv2.imshow("y", y[:,:,::-1])
    cv2.waitKey()