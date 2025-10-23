import torch.nn as nn
import torch.nn.functional as F


class PadTensor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        if sum(self.args[0]) == 0:
            return x
        else:
            return F.pad(x, *self.args, **self.kwargs)
