import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AdditiveAttention(nn.Module):
    """
    A general additive attention module.
    """
    def __init__(self, attention_query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim,
                                attention_query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(attention_query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, mask=None):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
            mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, attention_query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        weights = torch.matmul(temp, self.attention_query_vector)
        if mask is not None:
            weights[~mask] = float('-inf')
        weights = F.softmax(weights, dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
