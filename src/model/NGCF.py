import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NGCF(torch.nn.Module):
    def __init__(self, args):
        super(NGCF, self).__init__()
        self.args = args

    def forward(self):
        pass
