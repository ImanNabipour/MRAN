# Import libs
import os
from functools import wraps
import gc
import math
import random
from pathlib import Path      # 1.0.1
from datetime import datetime
from typing import List,Tuple
#import openpyxl

# scientific
import numpy as np    # 1.21.5
import matplotlib.pyplot as plt
import plotly.express as px # 5.9.0
from matplotlib import cm
import matplotlib          # 3.5.3
import tifffile as tiff   # 2021.7.2
import pandas as pd        # 1.3.4
from scipy.ndimage import zoom # 1.7.3
from sklearn.metrics import r2_score # 1.0.2
from sklearn.linear_model import LinearRegression

# torch
import torch   # 1.13
from torch.utils.data import Dataset
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.optim import (SGD,
                         Adam, )

from torchvision.transforms import (ToTensor,   # 0.13
                                    Compose,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    RandomRotation,
                                    Normalize,
                                    ToPILImage)
import time
#%%
from ResNeXt import  ResNeXt,ResNeXtBottleneck
#%%
base_dir = Path(r"D:\Iman_Nabipour\ToTuneCNNs2\Base-dir")# project root
base_output_dir = Path(r"D:\Iman_Nabipour\ToTuneCNNs2\Base-dir\cpt")
cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
output_dir = base_dir.joinpath("result").joinpath(cu)
output_dir.mkdir(parents=True, exist_ok=True)

# checkpoint_dir = "D:/Iman_Nabipour/ToTuneCNNs2/Base-dir/cpt/20230220-093548" # checkpoint for run on colab
checkpoint_dir =None     # "models" # new train va agar None nabud masire bala
if checkpoint_dir is None:
    checkpoint_dir = base_output_dir.joinpath(cu)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    has_cpt = False
else:
    checkpoint_dir = Path(checkpoint_dir)
    has_cpt = True

# new 3d images
## 1um_ High Resolution(HR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
base_path = "D:/Iman_Nabipour/ToTuneCNNs2/"
path1 = "Dataset_59/Dataset_Binary_59/"
path2 = "Dataset_59/Dataset_NormalizedDistField_59/"
um1_bin = Path(base_path + path1 + "1_HR_bin_59/HR_bin_Original_59")
um1_dist = Path(base_path + path2 + "1_HR_normdist_59/HR_Original_NormDist_59")
### Dilated
um1_bin_dilated=Path(base_path + path1 + "1_HR_bin_59/HR_bin_Dilated_59")
um1_dist_dilated = Path(base_path + path2 + "1_HR_normdist_59/HR_Dilated_NormDist_59")
### Eroded
um1_bin_eroded = Path(base_path + path1 + "1_HR_bin_59/HR_bin_Eroded_59")
um1_dist_eroded = Path(base_path + path2 + "1_HR_normdist_59/HR_Eroded_NormDist_59")
### opening
um1_bin_opening = Path(base_path + path1 + "1_HR_bin_59/HR_bin_opening_59")
um1_dist_opening = Path(base_path + path2 + "1_HR_normdist_59/HR_Opening_NormDist_59")

## 2um_ Midlle Resolution(MR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
um2_bin = Path(base_path + path1 + "2_MR_bin_59/MR_bin_Original_59")
um2_dist = Path(base_path + path2 + "2_MR_normdist_59/MR_Original_NormDist_59")
### Dilated
um2_bin_dilated=Path(base_path + path1 + "2_MR_bin_59/MR_bin_Dilated_59")
um2_dist_dilated = Path(base_path + path2 + "2_MR_normdist_59/MR_Dilated_NormDist_59")
### Eroded
um2_bin_eroded = Path(base_path + path1 + "2_MR_bin_59/MR_bin_Eroded_59")
um2_dist_eroded = Path(base_path + path2 + "2_MR_normdist_59/MR_Eroded_NormDist_59")
### opening
um2_bin_opening = Path(base_path + path1 + "2_MR_bin_59/MR_bin_opening_59")
um2_dist_opening = Path(base_path + path2 + "2_MR_normdist_59/MR_Opening_NormDist_59")

## 3um_ Low Resolution(LR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
um3_bin = Path(base_path + path1 + "3_LR_bin_59/LR_bin_Original_59")
um3_dist = Path(base_path + path2 + "3_LR_normdist_59/LR_Original_NormDist_59")
### Dilated
um3_bin_dilated=Path(base_path + path1 + "3_LR_bin_59/LR_bin_Dilated_59")
um3_dist_dilated = Path(base_path + path2 + "3_LR_normdist_59/LR_Dilated_NormDist_59")
### Eroded
um3_bin_eroded = Path(base_path + path1 + "3_LR_bin_59/LR_bin_Eroded_59")
um3_dist_eroded = Path(base_path + path2 + "3_LR_normdist_59/LR_Eroded_NormDist_59")
### opening
um3_bin_opening = Path(base_path + path1 + "3_LR_bin_59/LR_bin_Opening_59")
um3_dist_opening = Path(base_path + path2 + "3_LR_normdist_59/LR_Opening_NormDist_59")

# Labels file path:
lb_file_2 = Path(base_path + "Dataset_59/myLabels.xlsx")

## aggregation
um1_total_paths = (um1_bin,um1_dist,um1_bin_dilated,um1_dist_dilated,um1_bin_eroded,um1_dist_eroded,um1_bin_opening,um1_dist_opening)
um2_total_paths = (um2_bin,um2_dist,um2_bin_dilated,um2_dist_dilated,um2_bin_eroded,um2_dist_eroded,um2_bin_opening,um2_dist_opening)
um3_total_paths = (um3_bin,um3_dist,um3_bin_dilated,um3_dist_dilated,um3_bin_eroded,um3_dist_eroded,um3_bin_opening,um3_dist_opening)

## binary
um1_bin_paths = (um1_bin,um1_bin_dilated,um1_bin_eroded,um1_bin_opening)
um2_bin_paths = (um2_bin,um2_bin_dilated,um2_bin_eroded,um2_bin_opening)
um3_bin_paths = (um3_bin,um3_bin_dilated,um3_bin_eroded,um3_bin_opening)

## dist
um1_dist_paths = (um1_dist,um1_dist_dilated,um1_dist_eroded,um1_dist_opening)
um2_dist_paths = (um2_dist,um2_dist_dilated,um2_dist_eroded,um2_dist_opening)
um3_dist_paths = (um3_dist,um3_dist_dilated,um3_dist_eroded,um3_dist_opening)
#%% Utility

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flush_and_gc(f):
    @wraps(f)
    def g(*args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        return f(*args, **kwargs)

    return g
#%% Hyperparameters
EPOCHS = 100 #60
BATCH_SIZE =4    # 8
MOMENTUM = 0.8
WEIGHT_DECAY = 0 # 5e-2
N_FEATURE_MAP = 32
LR =  1e-3 # test range:(0.9, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7)
#%% Data Augmentation
random_h_flip_prob = .5
random_v_flip_prob = .5
random_degree_rotate_prob = 10.
n_channel = 120
#%% Runtime
train_size_per = .9
dev_size_per = .1
seed = 2021
save_iter = 5
n_worker = 4     #  2 for linux and 0 for windows
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
# Fix seed:
fix_all_seeds(seed=seed)
#%% Loss

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_hat, target):
        return torch.sqrt(self._mse(y_hat, target))


class RLoss(nn.Module):
    def __init__(self):
        super(RLoss, self).__init__()

    def forward(self, y_hat, target):
        n = target.shape[0]
        sum_reg = torch.sum(torch.pow(torch.sub(target, y_hat), 2))
        sum_tot = torch.sum(torch.pow(torch.sub(target, torch.mean(target)), 2))
        return torch.div(sum_reg, sum_tot+1e-14)


LOSS_FACTORY = {
    "mae": nn.L1Loss,
    "rmse": RMSELoss,
    "r": RLoss
}

#%% Dataloader
class Stone(Dataset):
    def __init__(self, images_dir: List[Path], label_xlx: Path, transformers):
        self._transformers = transformers
        self._ds_root = images_dir
        self._label_root = label_xlx
        self._lb = np.squeeze(pd.read_excel(str(self._label_root)).to_numpy(), axis=-1)
        lb = [self._lb for _ in range(len(images_dir))]
        self._lb = np.concatenate(lb, axis=0)
        self._f_list = []
        for d_path in self._ds_root:
            f_list = list(d_path.glob("*.tif"))
            # print('dpath',d_path,'flist',f_list)
            f_list.sort(key=lambda p: int(p.stem.split("-")[0]))
            self._f_list += f_list
        self.num = 0
        assert len(self._f_list) == len(self._lb)

    def __len__(self):
        return len(self._f_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_d_im = tiff.imread(str(self._f_list[idx]))
        #         n_d_im = zoom(n_d_im, (0.5, 0.5, 0.5))
        n_d_im = zoom(n_d_im, (0.4, 0.4, 0.4))
        #         print(n_d_im.shape)
        lb = torch.as_tensor(self._lb[idx] * 10 ** 15)
        #         print(lb.shape)
        n_d_im = (n_d_im - n_d_im.min()) / (n_d_im.max() - n_d_im.min())
        if self._transformers is not None:
            n_d_im = torch.unsqueeze(self._transformers(n_d_im.astype(np.float32)), 0).cuda()
            if self.num % 5 == 0:
                print('idx',self.num)
            self.num +=1

        return n_d_im, lb
#%% Proposed model

class Model3DV1(nn.Module):
    def __init__(self, n_channels, n_feature):
        super(Model3DV1, self).__init__()

        self._model = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_feature, kernel_size=3, bias=True),
            nn.InstanceNorm3d(n_feature),
            nn.CELU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(in_channels=n_feature, out_channels=n_feature * 2, kernel_size=3, bias=True),
            nn.InstanceNorm3d(n_feature * 2),
            nn.CELU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(in_channels=n_feature * 2, out_channels=n_feature * 4, kernel_size=3, bias=True),
            nn.InstanceNorm3d(n_feature * 4),
            nn.CELU(inplace=True),
            nn.MaxPool3d(kernel_size=3),

            nn.Conv3d(in_channels=n_feature * 4, out_channels=n_feature * 8, kernel_size=3, bias=True),
            nn.InstanceNorm3d(n_feature * 8),
            nn.CELU(inplace=True),
            nn.MaxPool3d(kernel_size=3),

            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=1),

            #             nn.Linear(in_features=65536, out_features=1),
            #             nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)
#%% test output shape

model = Model3DV1(n_channels=1, n_feature=N_FEATURE_MAP)
out = model(torch.randn([1,1,120,120,120]))
print(out.shape)
#%% Resnet model
# New resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

#%%
# How to instantiate and test output size:
import torch

# Instantiate the ResNet50 model
model = ResNet50(in_channels=1, num_classes=1)

# Create a random 3D image with dimensions 120x120x120
x = torch.randn((1, 1, 120, 120, 120))

# Pass the image through the model to obtain a regression output
y = model(x)

print(y.shape) # Output shape: torch.Size([1, 1])
#%% Densenet

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.InstanceNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.InstanceNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.CELU(inplace=True)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.relu(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.relu(out)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.InstanceNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.InstanceNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool3d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nFeatures, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv3d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.InstanceNorm3d(nChannels)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(nChannels, nFeatures)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.gap(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return (out)
    
#%%
densenet = DenseNet(growthRate = 6, depth = 25, reduction = 0.5, nFeatures = 1, bottleneck = True)
#%% SimpleNet
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()


        self.conv1 = nn.Conv3d(1,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.maxpool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.maxpool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.maxpool3 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout3d(p=0.1)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.maxpool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.maxpool5 = nn.MaxPool3d(2)
        self.dropout2 = nn.Dropout3d(p=0.1)

        self.conv6 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(512)

        self.conv7 = nn.Conv3d(512, 16, kernel_size=1)
        self.bn7 = nn.BatchNorm3d(16)
        self.dropout3 = nn.Dropout3d(p=0.1)

        self.fc1 = nn.Linear(16*27, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)

        out = F.elu(self.bn2(self.conv2(out)))
        out = self.maxpool2(out)

        out = F.elu(self.bn3(self.conv3(out)))
        out = self.dropout1(self.maxpool3(out))

        out = F.elu(self.bn4(self.conv4(out)))
        out = self.maxpool4(out)

        out = F.elu(self.bn5(self.conv5(out)))
        out = self.dropout2(self.maxpool5(out))

        out = F.elu(self.bn6(self.conv6(out)))

        out = self.dropout3(F.elu(self.bn7(self.conv7(out))))

        out = out.reshape(out.size(0), -1)

        out = F.elu(self.fc1(out))
        out = self.fc2(out)

        return out
#%% Test outout shape:
model=SimpleNet()
#model = DenseNet(growthRate=4, depth=20, reduction=0.25, nFeatures=1, bottleneck=False)  # default g.r=6, depth=25
print(model(torch.randn([1,1,120,120,120])))
#%% Transform:
def get_transforms(p_hor=.5, p_ver=.5, r_degree=10, mean=.5, std=.5, n_channel=120):
    print('n_ch',n_channel)
    return Compose([
        ToTensor(),
        # Normalize(mean=[mean] * n_channel, std=[std] * n_channel),
        RandomHorizontalFlip(p=p_hor),
        RandomVerticalFlip(p=p_ver),
#         RandomRotation(degrees=r_degree),
    ])

def get_simple_transformers(n_channel=300):
      return Compose([
            ToTensor(),
            Normalize(mean=[0.5] * n_channel, std=[0.5] * n_channel)
      ])
#%%
print(um1_bin_paths)
ts_ds = Stone(images_dir=um1_dist_paths,
                label_xlx=lb_file_2,
                transformers=None)
ts_data,ts_lb = ts_ds[np.random.randint(0,len(ts_ds))]
print("Permeability: ",ts_lb.item())
fig = px.imshow(ts_data, animation_frame=0, binary_string=True, labels=dict(animation_frame="scan"), height=600)
fig.show()

# um1_bin_paths   HR_test_path
#%% Trainer
class BaseTrain:
    def __init__(self, *args, **kwargs):
        super(BaseTrain).__init__(*args, **kwargs)

    @flush_and_gc
    def train_step(self, **kwargs):
        raise NotImplementedError

    @flush_and_gc
    def validation_step(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class TrainerV1(BaseTrain):
    def __init__(self, *args, **kwargs):
        super(TrainerV1, self).__init__()
        self._model = kwargs["model"]
        self._device = kwargs["device"]
        self._criterion = kwargs["criterion"]()
        self._opt = kwargs["opt"]
        self._scheduler = kwargs["scheduler"]

        self._model.to(self._device)
        self._opt = self._opt(self._model.parameters(),
                              lr=kwargs["lr"],
                              weight_decay=kwargs["weight_decay"])
        self._scheduler = self._scheduler(self._opt,
                                          step_size=10,
                                          gamma=0.1)

        self.preds = []
        self.targets = []

    @property
    def model(self):
        return self._model

    @flush_and_gc
    def train_step(self, **kwargs):
        data, lb = kwargs["batch"]
        data = data.to(self._device)
        lb = torch.unsqueeze(lb, dim=-1)
        lb = lb.float().to(self._device)
        loss = self._criterion(self._model(data), lb)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return loss.item()

    @flush_and_gc
    def validation_step(self, **kwargs):
        data, lb = kwargs["batch"]
        data = data.to(self._device)
        lb = torch.unsqueeze(lb, dim=-1)
        lb = lb.float().to(self._device)
        pred = self._model(data)
        for llb in lb:
            self.targets.append(llb.item())
        for lpred in pred:
            self.preds.append(lpred.item())
        return self._criterion(pred, lb).item()

    def train(self, **kwargs):
        print(f"-> Training is now starting")
        train_ld = kwargs["train_ld"]
        dev_ld = kwargs["dev_ld"]
        epoch_idx = kwargs["epoch_idx"]
        name = kwargs['name']
        model_phase = kwargs["model_phase"]
        cpt_dir = kwargs["cpk_dir"]
        save_iter = kwargs["save_iter"]
        con_results = kwargs["cont_results"]
        last_glob_t_loss = kwargs["last_glob_t_loss"]
        last_glob_d_loss = kwargs["last_glob_d_loss"]

        glob_t_loss = [] if last_glob_t_loss is None else last_glob_t_loss
        glob_d_loss = [] if last_glob_d_loss is None else last_glob_d_loss
        glob_r2_loss = []
        res_cont = {}
        res_cont["name"] = name
        epochs = kwargs["epochs"]
        glob_step = 0

        valid_data_cuda = [batch for batch in dev_ld]

        train_data_cuda = [batch for batch in train_ld]


        for epoch in range(epoch_idx, epochs):
            tm_t_loss = []
            prefix = f"[{epoch + 1}|{epochs}] epoch,"

            # train step
            self._model.train()
            randper = torch.randperm(len(train_data_cuda))
            #train_data_cuda = train_data_cuda[randper]
            self.targets = []
            self.preds = []

            for batch_idx in randper:#enumerate(train_data_cuda):
                batch = train_data_cuda[batch_idx]
                ls = self.train_step(batch=batch)
                tm_t_loss.append(ls)

                # show the result
                #if (glob_step + 1) % 10 == 0:
                print(f"{prefix} [{batch_idx}] batch, Loss -> train: {tm_t_loss[-1]}")

                if (glob_step + 1) % 2 == 0:
                    torch.save({
                        "model_phase": model_phase,
                        "epoch_idx": epoch,
                        "model": self.model.state_dict(),
                        "opt": self._opt.state_dict(),
                        "resluts": con_results,
                        "glob_t_loss": glob_t_loss,
                        "glob_d_loss": glob_d_loss
                    }, cpt_dir.joinpath("model.pt"))

                glob_step += 1

            glob_t_loss.append(np.array(tm_t_loss).mean())
            self._scheduler.step()

            # validation step
            tm_d_loss = []
            self._model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_data_cuda):
                    ls = self.validation_step(batch=batch)
                    tm_d_loss.append(ls)
            r2 = r2_score(np.array(self.targets), np.array(self.preds))

            glob_d_loss.append(np.array(tm_d_loss).mean())
            glob_r2_loss.append(r2)
            print(f"{prefix}, Loss -> train: {glob_t_loss[-1]}, dev: {glob_d_loss[-1]} , r2: {glob_r2_loss[-1]}")


        res_cont["results"] = {"train_loss": np.array(glob_t_loss), "dev_loss": np.array(glob_d_loss)}
        con_results.append(res_cont)

        torch.save({
            "model_phase": model_phase,
            "epoch_idx": epoch,
            "model": self.model.state_dict(),
            "opt": self._opt.state_dict(),
            "resluts": con_results,
            "glob_t_loss": glob_t_loss,
            "glob_d_loss": glob_d_loss
        }, cpt_dir.joinpath("model.pt"))
        return con_results

    def save(self, **kwargs):
        torch.save(self._model.state_dict(), str(kwargs["save_path"].joinpath("model.pt")))


def initialize_weights(m):
  if isinstance(m, nn.Conv3d):
      nn.init.normal_(m.weight.data)
      if m.bias is not None:
          nn.init.normal_(m.bias.data)
  elif isinstance(m, nn.BatchNorm3d):
      nn.init.normal_(m.weight.data)
      nn.init.normal_(m.bias.data)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.normal_(m.bias.data)

#%%

if __name__ == '__main__':


    if has_cpt:
        cpt_dic = torch.load(str(checkpoint_dir.joinpath("D:/Iman_Nabipour/ToTuneCNNs2/Base-dir/cpt/20230220-093548/model.pt")))


        idx = cpt_dic["model_phase"]
        epoch_idx = cpt_dic["epoch_idx"]
        model_state_dic = cpt_dic["model"]
        opt_state_dic = cpt_dic["opt"]
        results = cpt_dic["resluts"]
        last_glob_d_loss = cpt_dic["glob_d_loss"]
        last_glob_t_loss = cpt_dic["glob_t_loss"]
    else:
        idx = 0
        epoch_idx = 0
        results = []
        last_glob_d_loss = None
        last_glob_t_loss = None
#%%  path_sequence
    # The 3D images will be trained for regression and predicting Permeability sequentially in the order bellow:
    # path_sequence = (
    #               #("HR-Dist",um1_dist_paths),
    #               ("HR-BIN",um1_dist_paths),
    #               ("MR-BIN",um2_dist_paths),
    #               ("LR-BIN", um3_dist_paths),
    #               #("LR-BIN", um3_bin_paths)

    # )
    path_sequence = (
               ("HR-Dist",um1_dist_paths),
               ("IR-Dist",um2_dist_paths),
               ("LR-Dist",um3_dist_paths),
               ("HR-Bin",um1_bin_paths),            
               ("IR-Bin",um2_bin_paths),            
               ("LR-Bin",um3_bin_paths),
                )

#%%
    # define model
    print('Choose model for training ... ')
    print('1: Proposed model')
    print('2: Resnet model')
    print('3: Densenet model')
    print('insert 1, 2 or 3')
    #m = int(input())
    #if m == 1:
    #    print('you chose proposed model')
    #    model = Model3DV1(n_channels=1, n_feature=N_FEATURE_MAP)
    #elif m == 2:
    print('you chose Resnet model')
    model = ResNet50(in_channels=1, num_classes=1)
    # elif m == 3:
    #     print('you chose Densenet model')
    #model = DenseNet(growthRate = 2, depth = 20, reduction = 0.5, nFeatures = 1, bottleneck = False) # default g.r=6, depth=25
    # model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3],  block_inplanes= [128, 256, 512, 1024],n_classes=1,n_input_channels=1) #DenseNet(growthRate=6, depth=15, reduction=0.5, nFeatures=1, bottleneck=False)
    # print(model(torch.zeros(2,1,120,120,120)))
    # else:
    #     print('please choose a valid model')
#%%

    criterion = LOSS_FACTORY["mae"]
    opt = Adam
    scheduler = StepLR

    # For calc training time
    start = time.time()

    if has_cpt:
        model.load_state_dict(model_state_dic)
        print('load model')
    else:
        model.apply(initialize_weights)

    print(len(path_sequence),idx)

    for idx in range(len(path_sequence)):
        (name, p) = path_sequence[idx]

        print(name , p)
        dataset = Stone(images_dir=p,
                        label_xlx=lb_file_2,
                        transformers=get_transforms(p_hor=random_h_flip_prob,
                                                    p_ver=random_v_flip_prob,
                                                    r_degree=random_degree_rotate_prob,
                                                    n_channel=n_channel
                                                    ))
        train_size = int(len(dataset) * train_size_per)
        dev_size = len(dataset) - train_size
        train_set, dev_set = random_split(dataset, [train_size, dev_size])

        print(train_size, BATCH_SIZE, n_worker)

        train_loader = DataLoader(train_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0)#n_worker)
        #     print(len(train_loader))

        dev_loader = DataLoader(dev_set,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=0)#n_worker)

        print(f"[Trainer {idx + 1}] Name: {name}")
        trainer = TrainerV1(model=model,
                            criterion=criterion,
                            opt=opt,
                            scheduler=scheduler,
                            device=device,
                            weight_decay=WEIGHT_DECAY,
                            lr=LR,
                            momentum=MOMENTUM)

        results = trainer.train(train_ld=train_loader,
                                dev_ld=dev_loader,
                                epochs=EPOCHS,
                                epoch_idx=0,
                                #epoch_idx= epoch_idx if has_cpt else 0,
                                model_phase=idx,
                                cpk_dir=checkpoint_dir,
                                save_iter=save_iter,
                                cont_results=results,
                                name=name,
                                last_glob_t_loss=last_glob_t_loss,
                                last_glob_d_loss=last_glob_d_loss)
        has_cpt = False
        last_glob_t_loss = None
        last_glob_d_loss = None
        idx += 1
        model = trainer.model
        trainer.save(save_path=output_dir)

    stop = time.time()
    print(f"Training time: {stop - start} sec")
