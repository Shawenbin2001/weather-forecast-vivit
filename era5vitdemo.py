import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from vit_model import VideoViT
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import torch.nn.init as init
import numpy as np
import xarray as xr
import torch
from torch.utils.data import IterableDataset
import random

def gdl_loss(pred, target):
    """
    计算梯度差损失（Gradient Difference Loss, GDL）
    用于增强图像/视频预测的边缘和细节保留
    输入:
        pred: 预测张量，形状 (B, T, C, H, W)
        target: 目标张量，形状 (B, T, C, H, W)
    输出:
        gdl: 计算得到的GDL损失值
    """
    dx_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dx_target = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
    dy_target = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
    
    gdl_x = torch.abs(dx_pred - dx_target)
    gdl_y = torch.abs(dy_pred - dy_target)
    
    gdl = torch.mean(gdl_x) + torch.mean(gdl_y)
    return gdl


class ERA5Dataset(IterableDataset):
    def __init__(
        self,
        file_path,
        mean_path,
        std_path,
        in_len: int = 24,       # 每个样本的时间步长度（t）
        out_len: int = 6,       # 预测的时间步长度（t）
        stride: int = 6,         # 滑窗步长
        variables: list = ['u10', 'v10', 't2m'],  # 特征变量（对应c维度）
        shuffle: bool = False ,   # 是否打乱样本顺序
        split: str = 'train'  # 数据集划分（训练/验证/测试）

    ):
        super(ERA5Dataset, self).__init__()
        self.file_path = file_path
        self.mean_path = mean_path
        self.std_path = std_path
        self.in_len = in_len
        self.out_len = out_len
        self.stride = stride
        self.variables = variables
        self.shuffle = shuffle  # 控制是否打乱

        # 读取数据集
        if split == 'train':
            self.data = xr.open_dataset(self.file_path).isel(time=slice(0,15000)).fillna(0)
        elif split == 'val':
            self.data = xr.open_dataset(self.file_path).isel(time=slice(16000,17500)).fillna(0)
        
        self.mean = xr.open_dataset(self.mean_path).fillna(0)
        self.std = xr.open_dataset(self.std_path).fillna(0)

        t2m = self.data['t2m'].values
        u10 = self.data['u10'].values
        v10 = self.data['v10'].values
        t2m_mean = self.mean['t2m'].values
        u10_mean = self.mean['u10'].values
        v10_mean = self.mean['v10'].values
        t2m_std = self.std['t2m'].values
        u10_std = self.std['u10'].values
        v10_std = self.std['v10'].values

        nor_t2m = self.normalize(t2m, t2m_mean, t2m_std)
        nor_u10 = self.normalize(u10, u10_mean, u10_std)
        nor_v10 = self.normalize(v10, v10_mean, v10_std)

        self.nor_data = np.stack([nor_u10, nor_v10, nor_t2m], axis=-1)  # shape: (time, lat, lon, c)
        self.nor_data = np.transpose(self.nor_data, (0, 3, 1, 2))  # shape: (time, c, lat, lon)
        self.nor_data = self.nor_data[:,:,:160,:280]
        if np.isnan(self.nor_data).any():
            print("警告：归一化后数据包含NaN！")
        del t2m, u10, v10

        # ========== 预计算所有合法的起始索引 ==========
        self.valid_start_indices = self._get_valid_start_indices()

    def normalize(self, var, mean, std):
        return (var - np.mean(mean,axis=(0,1))) / (np.mean(std+1e-8,axis=(0,1)))
    
    # ========== 新增方法：计算合法起始索引（解决迭代范围和shuffle问题） ==========
    def _get_valid_start_indices(self):
        total_time_steps = self.nor_data.shape[0]
        # 核心：计算最大合法起始索引，避免索引越界
        max_start_idx = total_time_steps - self.in_len - self.out_len
        if max_start_idx < 0:
            raise ValueError(f"数据长度{total_time_steps}过短，无法满足in_len={self.in_len}+out_len={self.out_len}")
        
        # 生成所有合法的起始索引
        valid_indices = list(range(0, max_start_idx + 1, self.stride))
        
        # 正确的shuffle逻辑：打乱合法索引列表（而非随机生成索引）
        if self.shuffle:
            random.shuffle(valid_indices)
        
        return valid_indices
    
    def __len__(self):
        # ========== 关键修改2：len返回合法索引数量，保证与实际样本数一致 ==========
        return len(self.valid_start_indices)

    def __iter__(self):
        # ========== 关键修改3：遍历预计算的合法索引，避免越界和shuffle错误 ==========
        for start_idx in self.valid_start_indices:
            # 提取输入和输出样本（无需再判断shuffle，索引已提前打乱）
            sample_in = self.nor_data[start_idx:start_idx + self.in_len]  # shape: (seq_len, c, lat, lon)
            sample_out = self.nor_data[start_idx + self.in_len:start_idx + self.in_len + self.out_len]
        
            yield torch.tensor(sample_in, dtype=torch.float32), torch.tensor(sample_out, dtype=torch.float32)


def train_step(model, batch_in, batch_out, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(batch_in)  # shape: (batch_size, out_len, c, lat, lon)
    target = torch.cat([batch_in[:, 1:],batch_out], dim=1)
    loss = criterion(outputs, target)
    gdl = gdl_loss(outputs, target)
    loss = loss + 0.5 * gdl
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, dataloader, criterion,device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_in, batch_out in dataloader:
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)
            outputs = model(batch_in)[:,-1:]
            loss = criterion(outputs, batch_out)
            gdl = gdl_loss(outputs, batch_out)
            loss = loss + 0.5 * gdl
            total_loss += loss.item()
    return total_loss / dataloader.__len__()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def init_weights(m):
    """
    通用的权重初始化函数，自动识别层类型并初始化
    用法：model.apply(init_weights)
    """
    classname = m.__class__.__name__
    
    # 1. 线性层（Linear）初始化
    if classname.find('Linear') != -1:
        # Kaiming初始化（针对ReLU/LeakyReLU激活）
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        # 偏置初始化为0（避免偏移）
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    
    # 2. 卷积层（Conv1d/Conv2d，时序/空间特征提取）
    elif classname.find('Conv') != -1:
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

if __name__ == "__main__":
    train_dataset = ERA5Dataset(file_path='/home/data_public3/shawbdata/era5_train_data.nc',
                      mean_path='/home/data_public3/shawbdata/era5_mean.nc',
                      std_path='/home/data_public3/shawbdata/era5_std.nc',
                      in_len=3, out_len=1, stride=1, shuffle=True, split='train')
    val_dataset = ERA5Dataset(file_path='/home/data_public3/shawbdata/era5_train_data.nc',
                      mean_path='/home/data_public3/shawbdata/era5_mean.nc',
                      std_path='/home/data_public3/shawbdata/era5_std.nc',
                      in_len=3, out_len=1, stride=1, shuffle=False, split='val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = VideoViT(in_ch=3, embed_dim=256, patch_size=(8, 8), 
                     spatial_depth=4,  # 空间注意力层数
                     temporal_depth=4, # 时间注意力层数
                     num_heads=4)
    pram = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {pram/1000} k trainable parameters')
    device = torch.device('cuda:2')
    model = model.to(device)
    init_weights(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=15,  # 退火周期
        eta_min=1e-6                 # 最小学习率
    )
    
    num_epochs = 60
    train_losses = []
    val_losses = []
    bestval = np.inf
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_in, batch_out in train_dataloader:
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)
            loss = train_step(model, batch_in, batch_out, criterion, optimizer)
            cosine_scheduler.step()
            train_loss += loss
        avg_train_loss = train_loss / train_dataloader.__len__()
        if train_loss == np.nan:
            print("训练过程中出现NaN损失，终止训练")
            break
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        #if (epoch + 1) % 5 == 0:
        if val_loss < bestval:
            bestval = val_loss
            print(f"Validation loss improved, saving model at epoch {epoch+1}")
            save_model(model, f"era5vit_weight/videovit_best.pth")
    fig = plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('era5vit_loss_curve.png')
    plt.show()