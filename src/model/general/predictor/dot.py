import torch
import torch.nn as nn


class DotPredictor(nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def forward(self, first_vector, second_vector):
        '''
        Args:
            first_vector: (shape) batch_size, *,  dim
            second_vector: (shape) batch_size, *, dim
        Returns:
            (shape) batch_size, *
        '''
        return torch.mul(first_vector, second_vector).sum(dim=-1)


if __name__ == '__main__':
    first_vector = torch.rand(16, 10)
    second_vector = torch.rand(16, 10)
    dot_predictor = DotPredictor()
    print(dot_predictor(first_vector, second_vector))
