import torch
import torch.nn as nn
import dgl.function as fn
from dgl.utils import expand_as_pair


class LightGCNConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feature):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feature, graph)

            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1, ) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.srcdata['h'] = feat_src

            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1, ) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            return rst
