import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        raise NotImplementedError

    def forward(self, input, target):
        pass


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, positive, negative):
        return (1 - negative.view(positive.shape[0], -1) +
                positive.unsqueeze(1)).clamp(min=0).mean()
