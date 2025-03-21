import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A=None, num_nodes=25, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.constant_(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1, num_nodes, in_channels))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)
        self.relu = nn.ReLU()

        index_array = np.empty(num_nodes * in_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * num_nodes)
        self.register_buffer('shift_in', torch.from_numpy(index_array))

        index_array = np.empty(num_nodes * out_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * num_nodes)
        self.register_buffer('shift_out', torch.from_numpy(index_array))

    def forward(self, x0, edge_index = None):
        x0_proc = x0.permute(0, 3, 1, 2).contiguous()  # (n, c, t, v)
        n, c, t, v = x0_proc.size()
        x = x0_proc.permute(0, 2, 3, 1).contiguous()  # (n, t, v, c)
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('nwc,cd->nwd', x, self.Linear_weight).contiguous()
        x = x + self.Linear_bias
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2).contiguous()
        shortcut = self.down(x0_proc)
        x = x + shortcut
        x = self.relu(x)
        return x
