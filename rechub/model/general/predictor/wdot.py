import torch
import torch.nn as nn


class WDotPredictor(nn.Module):
    def __init__(self, in_dims, out_dim):
        super().__init__()
        assert len(in_dims) == 2
        self.first_linear = nn.Linear(in_dims[0], out_dim)
        self.second_linear = nn.Linear(in_dims[1], out_dim)

    def forward(self, first_vector, second_vector):
        '''
        Args:
            first_vector: (shape) batch_size, *,  dim
            second_vector: (shape) batch_size, *, dim
        Returns:
            (shape) batch_size, *
        '''
        return torch.mul(
            self.first_linear(first_vector),
            self.second_linear(second_vector),
        ).sum(dim=-1)


if __name__ == '__main__':
    first_vector = torch.rand(16, 10)
    second_vector = torch.rand(16, 12)
    wdot_predictor = WDotPredictor((10, 12), 10)
    print(wdot_predictor(first_vector, second_vector))
