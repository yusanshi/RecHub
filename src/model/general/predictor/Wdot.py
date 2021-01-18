import torch
import torch.nn as nn


class WDotPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(WDotPredictor, self).__init__()
        self.in_dim = in_dim
        if isinstance(in_dim, tuple):
            assert len(in_dim) == 2
            self.first_linear = nn.Linear(in_dim[0], out_dim)
            self.second_linear = nn.Linear(in_dim[1], out_dim)
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, first_vector, second_vector):
        '''
        Args:
            first_vector: (shape) batch_size, *,  dim
            second_vector: (shape) batch_size, *, dim
        Returns:
            (shape) batch_size, *
        '''
        if isinstance(self.in_dim, tuple):
            return torch.mul(
                self.first_linear(first_vector),
                self.second_linear(second_vector),
            ).sum(dim=-1)
        else:
            return torch.mul(
                self.linear(first_vector),
                self.linear(second_vector),
            ).sum(dim=-1)


if __name__ == '__main__':
    first_vector = torch.rand(16, 10)
    second_vector = torch.rand(16, 10)
    wdot_predictor = WDotPredictor(10, 10)
    print(wdot_predictor(first_vector, second_vector))
