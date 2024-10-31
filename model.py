import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
        self.attention = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[0],layer_sizes[0])
        )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x

class Attention(nn.Module):
    def __init__(self, input_dims=64,hidden_dims=64,head=1, dropout=0.0):
        super().__init__()
        self.q_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
        self.k_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
        self.head=head
        self.num=hidden_dims//head
        self.input_dims=input_dims


    def forward(self,q,k):
        query=self.q_linear(q)
        key=self.k_linear(k)

        query=query.view(-1,self.head,self.num).transpose(0, 1)
        key=key.view(-1,self.head,self.num).transpose(0, 1)

        attn_matrix=torch.bmm(query,key.transpose(1, 2))
        attn_matrix=torch.sum(attn_matrix,dim=0)

        return attn_matrix

class Q_Net(nn.Module):
    def __init__(self, worker_state_size, order_state_size, hidden_dim=64, head=1, arl=True, dropout=0.0):
        super().__init__()
        self.worker_net = MLP([worker_state_size,hidden_dim,hidden_dim], arl = arl, dropout = dropout)
        self.order_net = MLP([order_state_size,hidden_dim,hidden_dim], arl = arl, dropout = dropout)
        self.attention = Attention(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head,dropout=dropout)

    def forward(self,x_worker,x_order):
        x_worker, x_order = x_worker.float(), x_order.float()
        x_worker = self.worker_net(x_worker)
        x_order = self.order_net(x_order)
        q_value = self.attention(x_worker,x_order)
        return q_value