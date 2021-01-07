import torch
import torch.nn as nn
import dgl


class NGCF(nn.Module):
    def __init__(self):
        super(NGCF, self).__init__()

    def forward(self, g, inputs):
        pass


if __name__ == '__main__':
    graph = dgl.graph((torch.randint(10, (20, )), torch.randint(10, (20, ))))
    graph = dgl.to_simple(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    model = NGCF(16, 16, 16).to(device)
    inputs = torch.rand(10, 16).to(device)
    print(model(graph, inputs))
