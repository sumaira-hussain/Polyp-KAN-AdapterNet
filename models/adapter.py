import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, in_channels, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(in_channels * mlp_ratio)
        self.act = act_layer()
        # self.D_fc1 = nn.Linear(in_channels, D_hidden_features)
        # self.D_fc2 = nn.Linear(D_hidden_features, in_channels)
        self.D_fc1 = nn.Conv2d(in_channels, D_hidden_features, kernel_size=1)
        self.D_fc2 = nn.Conv2d(D_hidden_features, in_channels, kernel_size=1)

    def forward(self, x):
        # x is (BT, HW+1, D)
        identity = x  # Preserve original input for skip connection
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return x + identity if self.skip_connect else x  # Maintain spatial dimensions
        # if self.skip_connect:
        #     x = x + xs
        # else:
        #     x = xs
        # return x