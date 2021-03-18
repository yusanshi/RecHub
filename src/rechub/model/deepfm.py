import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DeepFM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self):
        pass
