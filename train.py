import torch
from model import SRCNN
import pickle as pkl
from tqdm import tqdm

import numpy as np

with open("dataset_hr.pkl","rb") as f:
    dataset_hr = pkl.load(f)

with open("dataset_lr.pkl","rb") as f:
    dataset_lr = pkl.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

best_avg_psnr = 0

# def eval(model):
#     model.eval()
#     model.train()

model = SRCNN().to(device)
opt = torch.optim.Adam(model.parameters(),1e-4)
loss_fn = torch.nn.MSELoss()
for epoch in range(50):
    print("epoch", epoch)
    for i in tqdm(range(len(dataset_hr)//100)):
        indices = np.random.randint(0, len(dataset_hr),(100,))
        hr_batch = dataset_hr[indices]
        lr_batch = dataset_lr[indices]
        hr_batch = torch.from_numpy(hr_batch).permute(0,3,1,2).float().to(device)
        lr_batch = torch.from_numpy(lr_batch).permute(0,3,1,2).float().to(device)
        # print(hr_batch.shape)
        # exit()
        y = model(lr_batch)
        loss = loss_fn(y, hr_batch)
        opt.zero_grad()
        # print(loss)
        loss.backward()
        opt.step()
        # exit()
        # eval(model)
    torch.save(model.state_dict(),"model.th")
