import os
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
import torchvision

logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################


def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(
                    f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    # if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())
    #    model.to(gpu_ids[0])
    #    model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
# class CustomTensorDataset(Dataset):
#     """TensorDataset with support of transforms."""
#     def __init__(self, tensors, transform=None):
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#         self.tensors = tensors
#         self.transform = transform

#     def __getitem__(self, index):
#         x = self.tensors[0][index]
#         y = self.tensors[1][index]
#         if self.transform:
#             x = self.transform(x.numpy().astype(np.uint8))
#         return x, y

#     def __len__(self):
#         return self.tensors[0].size(0)

'''
class MyDataset(Dataset):

  def __init__(self, file_name=None, transform=None):

    df_x = pd.read_csv("/Users/rishitas/Downloads/{}_x".format(file_name))
    df_y = pd.read_csv("/Users/rishitas/Downloads/{}_y".format(file_name))

    selected_cols = ['avgMeasuredTime', 'extID', 'medianMeasuredTime', 'vehicleCount', '_id', 'REPORT_ID', 'time', 'month', 'day',
       'year', 'hour', 'minute', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
       'minute_sin', 'minute_cos']
    x = df_x.values
    y = df_y.avgSpeed.values

    self.x_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y, dtype=torch.float32)
    self.transform = transform

  def get_dataset():
    return self.x_train, self.y_train

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.x_train[idx], self.y_train[idx]

  def data(self, idx):
    return self.x_train[idx], self.y_train[idx]

'''
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors_x, tensors_y, transform=None):
    # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors_x = tensors_x
        self.tensors_y = tensors_y
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors_x[index]
        y = self.tensors_y[index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors_x.size(0)

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    df = pd.read_csv("/Users/rishitas/Downloads/MTECHProject/Combine_train.csv")
    selected_cols = ['avgMeasuredTime', 'extID', 'medianMeasuredTime', 'vehicleCount',
                     '_id', 'month', 'day',
            'year', 'hour', 'minute', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos']
    #train = torch.tensor(train[[selected_cols]].values.astype(np.float32))
    #train_target = torch.tensor(train['avgSpeed'].values.astype(np.float32))
    
    local_datasets = []
    for i in df['REPORT_ID'].unique():
        #print(df[selected_cols].columns)
        local_datasets.append(CustomTensorDataset(
          torch.tensor(df[df['REPORT_ID']==i][selected_cols].head(1).values.astype(np.float32)),
          torch.tensor(df[df['REPORT_ID']==i]['avgSpeed'].head(1).values.astype(np.float32))))

    df_test = pd.read_csv("/Users/rishitas/Downloads/MTECHProject/Combine_test.csv")
    test_dataset = CustomTensorDataset(
          torch.tensor(df[selected_cols].values.astype(np.float32)),
          torch.tensor(df['avgSpeed'].values.astype(np.float32)))

    return local_datasets, test_dataset
