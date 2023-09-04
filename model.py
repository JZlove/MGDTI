import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_score
from torch.nn import init

import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init

class Predict(nn.Module):
    def __init__(self, hidden_dim, vars_dict, drop_rate=0.2) -> None:
        super(Predict, self).__init__()
        self.mlp = MLP(hidden_dim*2, 1, vars_dict, drop_rate=drop_rate)

    def forward(self, smi_common, fas_common, vars_list = None):
        commom = self.mlp(torch.cat((smi_common, fas_common), 1), vars_list)
        return commom


class MLP(nn.Module):
    def __init__(self, hidden_dim, out_dim, vars_dict, drop_rate):
        super(MLP, self).__init__()
        self.drop_rate = drop_rate
        self.vars = vars_dict
        self.vars["ml_fc_w1"] = self.get_initialed_para_matrix(hidden_dim, hidden_dim//2)
        self.vars["ml_fc_b1"] = self.get_zero_para_bias(hidden_dim//2)

        self.vars["ml_fc_w2"] = self.get_initialed_para_matrix(hidden_dim//2, hidden_dim//4)
        self.vars["ml_fc_b2"] = self.get_zero_para_bias(hidden_dim//4)

        self.vars["ml_fc_w3"] = self.get_initialed_para_matrix(hidden_dim//4, out_dim)
        self.vars["ml_fc_b3"] = self.get_zero_para_bias(out_dim)  
        # for model in self.linear:
        #     if isinstance(model, nn.Linear):
        #         nn.init.xavier_normal_(model.weight, gain=1.414)
    def get_initialed_para_matrix(self, in_num, out_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w
    
    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))
    
    def forward(self, x, vars_dict = None):
        if vars_dict is None:
            vars_dict = self.vars
        bs = len(x)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.dropout(x, self.drop_rate)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.dropout(x, self.drop_rate)
        out = torch.sigmoid(F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3']))

        return out.reshape(bs)


class AGTLayer(nn.Module):
    def __init__(self, vars_list, index, embeddings_dimension, nheads=2, att_dropout=0.3, emb_dropout=0.3, temper=1.0, rl=False, rl_dim=4, beta=1):

        super(AGTLayer, self).__init__()

        self.vars = vars_list
        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads

        self.leaky = nn.LeakyReLU(0.01)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.vars[f"GT_linear_l_w_{index}"] = self.get_initialed_para_matrix(
            self.embeddings_dimension, self.head_dim * self.nheads)

        self.vars[f"GT_linear_r_w_{index}"] = self.get_initialed_para_matrix(
            self.embeddings_dimension, self.head_dim * self.nheads)

        self.vars[f"GT_att_l_{index}"] = self.get_initialed_para_matrix(self.head_dim, 1)
        self.vars[f"GT_att_r_{index}"] = self.get_initialed_para_matrix(self.head_dim, 1)
        self.vars[f"linear_final_{index}"] = self.get_initialed_para_matrix(
            self.head_dim * self.nheads, self.embeddings_dimension)

        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)
        self.LN = nn.LayerNorm(embeddings_dimension)

    def get_initialed_para_matrix(self, in_num, out_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w
    
    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))
    
    def forward(self, h, index, vars_list = None):
        if vars_list is None:
            vars_list = self.vars
        batch_size = h.size()[0]
        fl = F.linear(h, vars_list[f"GT_linear_l_w_{index}"], None).reshape(
            batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = F.linear(h, vars_list[f"GT_linear_r_w_{index}"], None).reshape(
            batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        score = F.linear(self.leaky(fl), vars_list[f"GT_att_l_{index}"], None) + \
            F.linear(self.leaky(fr), vars_list[f"GT_att_r_{index}"], None).permute(0, 1, 3, 2)
        score = score / self.temper

        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)

        context = score @ fr
        h_sa = context.transpose(1, 2).reshape(
            batch_size, -1, self.head_dim * self.nheads)
        fh = F.linear(h_sa, vars_list[f"linear_final_{index}"], None)
        fh = self.dropout2(fh)
        h = self.LN(h + fh)
        return h

class fcList(nn.Module):
    def __init__(self, vars_dict, in_dim, out_dim, index) -> None:
        super().__init__()
        self.vars = vars_dict
        self.vars[f'fc_list_w_{index}'] = self.get_initialed_para_matrix(in_dim, out_dim)
        self.vars[f'fc_list_b_{index}'] = self.get_zero_para_bias(out_dim)

    def get_initialed_para_matrix(self, in_dim, out_dim):
        w = torch.nn.Parameter(torch.ones([out_dim, in_dim]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, dim):
        return torch.nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, index, vars_dict = None):
        if vars_dict is None:
            vars_dict = self.vars
        out = F.linear(x, vars_dict[f'fc_list_w_{index}'], vars_dict[f'fc_list_b_{index}'])
        return out

class Proj(nn.Module):
    def __init__(self, vars_dict, in_dim, out_dim, index) -> None:
        super().__init__()
        self.vars = vars_dict
        self.vars[f'proj_w_0_{index}'] = self.get_initialed_para_matrix(in_dim, out_dim)
        self.vars[f'proj_b_0_{index}'] = self.get_zero_para_bias(out_dim)
        self.vars[f'proj_w_1_{index}'] = self.get_initialed_para_matrix(in_dim, out_dim)
        self.vars[f'proj_b_1_{index}'] = self.get_zero_para_bias(out_dim)

    def get_initialed_para_matrix(self, in_dim, out_dim):
        w = torch.nn.Parameter(torch.ones([out_dim, in_dim]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, dim):
        return torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x, index, vars_dict = None):
        if vars_dict is None:
            vars_dict = self.vars
        x = F.linear(x, vars_dict[f'proj_w_0_{index}'], vars_dict[f'proj_b_0_{index}'])
        x = F.elu(x)
        out = F.linear(x, vars_dict[f'proj_w_1_{index}'], vars_dict[f'proj_b_1_{index}'])
        return out
     
class mainModel(nn.Module):
    def __init__(self, g, input_dimensions = None, embeddings_dimension=256, num_layers=4, num_gnns=2, nheads=2, dropout=0.2,  temper=1.0, num_type=4, beta=1):
        super(mainModel, self).__init__()

        self.g = g
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_gnns = num_gnns
        self.nheads = nheads

        self.vars = torch.nn.ParameterDict()

        self.fc_list = [fcList(self.vars, in_dim, embeddings_dimension, index) for index, 
                        in_dim in enumerate(input_dimensions)]
        
        self.proj_d = Proj(self.vars, embeddings_dimension, embeddings_dimension, 'd')
        self.proj_p = Proj(self.vars, embeddings_dimension, embeddings_dimension, 'p')

        self.predict = Predict(embeddings_dimension, self.vars)
        self.dropout = dropout
        self.GCNLayers = nn.ModuleList()
        self.GTLayers_dg = nn.ModuleList()
        self.GTLayers_pt = nn.ModuleList()


        for layer in range(self.num_gnns):
            self.vars[f"gcn_w_{layer}"] = self.get_initialed_para_matrix(embeddings_dimension, embeddings_dimension)
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, weight=False, activation=F.relu, allow_zero_in_degree=True))
        self.Transformer = []
        for layer in range(self.num_layers):
            self.GTLayers_dg.append(
                AGTLayer(self.vars, f"dg_{layer}", self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True, rl_dim=num_type, beta=beta))

        for layer in range(self.num_layers):
            self.GTLayers_pt.append(
                AGTLayer(self.vars, f"pt_{layer}", self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True, rl_dim=num_type, beta=beta))

        self.Transformer.append(self.GTLayers_dg)
        self.Transformer.append(self.GTLayers_pt)
        self.Drop = nn.Dropout(self.dropout)

    def get_initialed_para_matrix(self, in_dim, out_dim):
        w = torch.nn.Parameter(torch.ones([out_dim, in_dim]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, dim):
        return torch.nn.Parameter(torch.zeros(dim))
    
    def graph_transformer(self, r, gh, seqs, node_type, var_list = None):
        h = gh[seqs]
        r = r[seqs]
        type_name = "dg" if node_type == 0 else "pt"
        for layer in range(self.num_layers):
            h = self.Transformer[node_type][layer](h, f"{type_name}_{layer}", var_list)
        return h[:, 0, :]

    def forward(self, features_list, dg_seqs, pt_seqs, type_emb, node_type, vars_list = None):
        if vars_list is None:
            vars_list = self.vars
        h = []
        for index, (fc, feature) in enumerate(zip(self.fc_list, features_list)):
            h.append(fc(feature, index, vars_list))

        gh = torch.cat(h, 0)
        r = type_emb[node_type]
        for layer in range(self.num_gnns):
            gh = self.GCNLayers[layer](self.g, gh, weight=vars_list[f'gcn_w_{layer}'])
            gh = self.Drop(gh)

        dg_emb = self.graph_transformer(r, gh, dg_seqs, 0, vars_list)
        pt_emb = self.graph_transformer(r, gh, pt_seqs, 1, vars_list)

        dg_emb = self.proj_d(dg_emb, 'd', vars_list)
        pt_emb = self.proj_p(pt_emb, 'p', vars_list)

        pred = self.predict(dg_emb, pt_emb, vars_list)

        return pred
    
    def update_parameters(self):
        return self.vars