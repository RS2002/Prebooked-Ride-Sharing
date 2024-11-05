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


class LSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size, hidden_size // 2, output_size], dropout=dropout)

    def forward(self,x,valid_index=None):
        self.lstm.flatten_parameters()
        x,_ = self.lstm(x)
        x = torch.concat([torch.zeros(x.shape[0],1,x.shape[2]).to(x.device),x],dim=1)
        if valid_index is not None:
            x = x[torch.arange(x.size(0)),valid_index]
        else:
            x = x[:, -1, :]
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size // 2, batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size // 2, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size, hidden_size // 2, output_size], dropout=dropout)

    def forward(self,x,valid_index=None):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x1,_ = self.lstm1(x)
        x1 = torch.concat([torch.zeros(x1.shape[0],1,x1.shape[2]).to(x1.device),x1],dim=1)
        if valid_index is not None:
            x1 = x1[torch.arange(x1.size(0)),valid_index]
            x2 = x.clone()
            for i in range(x2.shape[0]):
                end_index = valid_index[i].item()
                x2[i, :end_index] = x[i, :end_index].flip(dims=(0,))
        else:
            x1 = x1[:, -1, :]
            x2 = torch.flip(x, [1])
        x2, _ = self.lstm2(x2)
        x2 = torch.concat([torch.zeros(x2.shape[0],1,x2.shape[2]).to(x2.device),x2],dim=1)
        if valid_index is not None:
            x2 = x2[torch.arange(x2.size(0)),valid_index]
        else:
            x2 = x2[:, -1, :]
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
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


class Worker_Net(nn.Module):
    def __init__(self, state_size, order_size, output_dim=64, bi_direction=False, dropout=0.0):
        super().__init__()
        if bi_direction:
            self.lstm = BiLSTM(order_size,32, dropout=dropout)
        else:
            self.lstm = LSTM(order_size,32, dropout=dropout)
        self.encode = MLP([state_size,16,32], arl=True, dropout=dropout)
        self.mlp = MLP([64,64,32,output_dim], dropout=dropout)

    def forward(self,x_state,x_order,order_num=None):
        x_order = self.lstm(x_order,order_num)
        x_state = self.encode(x_state)
        y = self.mlp(torch.concat([x_state,x_order],dim=-1))
        return y

class Order_Net(nn.Module):
    def __init__(self, state_size, output_size=64, dropout=0.0):
        super().__init__()
        self.model = MLP([state_size,16,32,output_size], arl=True, dropout=dropout)

    def forward(self,x):
        y = self.model(x)
        return y


class Q_Net(nn.Module):
    def __init__(self, state_size=11, history_order_size=4, current_order_size=5, hidden_dim=64, head=1, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.attention = Attention(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)

    def forward(self,order,x_state,x_order,order_num=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()

        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)
        q_matrix = self.attention(worker,order)

        return q_matrix