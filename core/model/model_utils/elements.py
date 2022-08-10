import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

BN = True
# BN = False

class BipartiteLayer(nn.Module):
    def __init__(self, nhid, mlp_num_layers=2):
        super().__init__()
        self.mlp1 = MLP(nhid, nhid, mlp_num_layers, with_final_activation=True)
        self.mlp2 = MLP(nhid, nhid, mlp_num_layers, with_final_activation=True)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
    
    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.eps.data.fill_(0)

    def forward(self, x, edge_index):
        left = scatter(x[edge_index[1]], edge_index[0], dim=0, reduce='add')
        left = self.mlp1(left)
        right = scatter(left[edge_index[0]], edge_index[1], dim=0, reduce='add')
        right = (1 + self.eps)*x + self.mlp2(right)
        return right

class MultiBipartitesLayer(nn.Module):
    def __init__(self, nhid, num_bipartites, mlp_num_layers=2):
        super().__init__()
        self.num_bipartites = num_bipartites
        self.backward_layers = nn.ModuleList(MLP(nhid, nhid, mlp_num_layers, with_final_activation=True) for _ in range(num_bipartites))
        self.backward_eps = torch.nn.Parameter(torch.Tensor([0]*num_bipartites))
        self.forward_layers = nn.ModuleList(MLP(nhid, nhid, mlp_num_layers, with_final_activation=True) for _ in range(num_bipartites))
        self.forward_eps = torch.nn.Parameter(torch.Tensor([0]*num_bipartites))

    def reset_parameters(self):
        for layer in self.backward_layers:
            layer.reset_parameters()
        for layer in self.forward_layers:
            layer.reset_parameters()

    def forward(self, xs, k_batch, bipartites_list): 
        # from right to left and then left to right
        for i in reversed(range(self.num_bipartites)):
            edge_index = bipartites_list[i]
            x_right = xs[k_batch == i+1]
            x_left = xs[k_batch == i]
            aggregated = scatter(x_right[edge_index[1]], edge_index[0], dim=0, reduce='add')
            xs[k_batch == i] = (1 + self.backward_eps[i])*x_left + self.backward_layers[i](aggregated)
        for i in range(self.num_bipartites):
            edge_index = bipartites_list[i]
            x_right = xs[k_batch == i+1]
            x_left = xs[k_batch == i]
            aggregated = scatter(x_left[edge_index[0]], edge_index[1], dim=0, reduce='add')
            xs[k_batch == i+1] = (1 + self.forward_eps[i])*x_right + self.forward_layers[i](aggregated)
        return xs


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=40): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def init_constant(self, value=0):
        for embedding in self.embeddings:
            embedding.weight.data.fill_(value)

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True, n_hid=None):
        super().__init__()
        if n_hid is None:
            n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)  

        # if self.residual:
        #     x = x + previous_x  
        return x 


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_model, heads=1, dropout=0):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_in, d_model)
        self.v_linear = nn.Linear(d_in, d_model)
        self.k_linear = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, self.h, self.d_k, 1)
        q = self.q_linear(q).view(bs, self.h, self.d_k, 1)
        v = self.v_linear(v).view(bs, self.h, self.d_k, 1)
        
        scores = attention(q, k, v, self.d_k, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.contiguous().view(bs, self.d_model)
        output = self.out(concat)
        return output

import math
def attention(q, k, v, d_k, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output