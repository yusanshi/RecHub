import torch
import torch.nn as nn


class DNNPredictor(nn.Module):
    def __init__(self, dims):
        super(DNNPredictor, self).__init__()
        assert len(dims) >= 2 and dims[-1] == 1

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.dnn = nn.Sequential(*layers)

    def forward(self, first_vector, second_vector):
        '''
        Args:
            first_vector: (shape) batch_size, *,  dim
            second_vector: (shape) batch_size, *, dim
        Returns:
            (shape) batch_size, *
        '''
        return self.dnn(torch.cat((first_vector, second_vector),
                                  dim=-1)).squeeze(dim=-1)


if __name__ == '__main__':
    first_vector = torch.rand(16, 8, 10)
    second_vector = torch.rand(16, 8, 10)
    dnn_predictor = DNNPredictor([20, 4, 1])
    print(dnn_predictor)
    print(dnn_predictor(first_vector, second_vector))
