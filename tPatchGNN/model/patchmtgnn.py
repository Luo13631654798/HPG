import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange, repeat

import lib.utils as utils
from lib.encoder_decoder import *
from lib.evaluation import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
import math
import lib.utils as utils
from torch import Tensor
from torch_scatter import scatter
import torch
# from time import time
from torch_geometric.utils.num_nodes import maybe_num_nodes

def softmax(src, index):
    # norm
    N = maybe_num_nodes(index)

    # out = src
    # out = out.exp()
    # out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    # local  max
    # local_out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    # local_out = local_out.exp()
    # local_out_sum = scatter(local_out, index, dim=0, dim_size=N, reduce='sum')[index]
    #
    # # global  max
    global_out = src - src.max()
    global_out = global_out.exp()
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    #
    # a = out / (out_sum + 1e-8)
    # b = local_out / (local_out_sum + 1e-16)
    c =  global_out / (global_out_sum + 1e-16)
    # eps = 0.0000001
    # print((torch.abs(a-c)>eps).sum())
    # print((torch.abs(a-b)>eps).sum())
    return c


class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor(
            [1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)])  # [20]
        self.div_term = torch.reshape(self.div_term, (1, -1))
        self.div_term = nn.Parameter(self.div_term, requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1, 1)
        t = t * 200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t, self.div_term)
        position_term[:, 0::2] = torch.sin(position_term[:, 0::2])
        position_term[:, 1::2] = torch.cos(position_term[:, 1::2])

        return position_term


class UA_GTrans(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, **kwargs):
        super(UA_GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_k // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        # self.w_k_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for i in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for i in range(self.n_heads)])
        # self.w_v_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])

        # self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_transfer = nn.ModuleList([nn.Linear(self.d_input + 1, self.d_k, bias=True) for i in range(self.n_heads)])

        # initiallization
        utils.init_network_weights(self.w_k_list)
        # utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list)
        # utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)

        # Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        # Normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value, time_nodes, edge_uncertainty):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value, edge_uncertainty=edge_uncertainty, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_uncertainty):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            k_linear = self.w_k_list[i]
            # k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i]
            # v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1, 1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer, k_linear, q_linear, x_i , edge_uncertainty)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention_norm = softmax(attention, edge_index_i)  # [4,1]

            M = torch.pow(self.alpha, edge_uncertainty)
            attention_norm = attention_norm * M.unsqueeze(-1)
            # attention_norm = torch.softmax(attention)  # [4,1]
            # attention_norm = attention  # [4,1]
            # attention_norm = (attention - torch.min(attention)) / (torch.max(attention) - torch.min(attention))

            sender = v_linear(x_j_transfer)
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, w_q, x_i, edge_uncertainty):
        x_i = w_q(x_i)  # receiver #[num_edge,d*heads]

        # wraping k

        sender = w_k(x_j_transfer)
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # edge_uncertainty = (edge_uncertainty - torch.min(edge_uncertainty)) / (torch.max(edge_uncertainty) - torch.min(edge_uncertainty))
        # edge_uncertainty = 2 * edge_uncertainty - 1
        # M = 1 / torch.exp(edge_uncertainty)
        # M = torch.pow(self.alpha, edge_uncertainty)
        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class UA_HPG(nn.Module):
    def __init__(self, args, supports=None):

        super(UA_HPG, self).__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim - 1)

        self.obs_enc = nn.Linear(1, args.hid_dim)

        d_model = args.hid_dim

        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)


        ### Encoder output layer ###
        self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))

        ### Decoder ###
        self.decoder = nn.Sequential(
            # nn.Linear(enc_dim+args.te_dim, args.hid_dim),
            nn.Linear(nodevec_dim + enc_dim + 2 * args.te_dim, 4 * args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * args.hid_dim, 2 * args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * args.hid_dim, 1)
        )

        # self.decoder = nn.Sequential(
        # 	# nn.Linear(enc_dim+args.te_dim, args.hid_dim),
        # 	nn.Linear(nodevec_dim + enc_dim + 2 * args.te_dim, args.hid_dim),
        # 	nn.ReLU(inplace=True),
        # 	nn.Linear(args.hid_dim, args.hid_dim),
        # 	nn.ReLU(inplace=True),
        # 	nn.Linear(args.hid_dim, 1)
        # )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        for i in range(layer_nums):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            # print(cur_adj.shape)

            edge_ind = torch.where(cur_adj == 1)
            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            # edge_mask = torch.where(edge_time <= 0)[0]
            # edge_index = edge_index[:, edge_mask]
            # edge_time = edge_time[edge_mask]

            # edge_same = (edge_time < 100).float()
            # edge_same = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same = torch.ones_like(edge_time).to(edge_time.device)
            edge_uncertainty = cur_x_uncertainty[source_nodes] - cur_x_uncertainty[target_nodes]
            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_uncertainty)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            x_uncertainty_per_patch = torch.sum(x_uncertainty, dim=3)

            if M > 1 and M % 2 != 0:
                node_state_sum_per_patch[:, :, -2, :] += node_state_sum_per_patch[:, :, -1, :]
                node_state_sum_per_patch = node_state_sum_per_patch[:, :, :-1, :]

                # x[:, :, -2] = torch.max(x[:, :, -2], x[:, :, -1])
                # x = x[:, :, :-1]

                obs_num_per_patch[:, :, -2, :] += obs_num_per_patch[:, :, -1, :]
                obs_num_per_patch = obs_num_per_patch[:, :, :-1, :]

                x_time_per_patch[:, :, -2, :] += x_time_per_patch[:, :, -1, :]
                x_time_per_patch = x_time_per_patch[:, :, :-1, :]

                x_uncertainty_per_patch[:, :, -2] += x_uncertainty_per_patch[:, :, -1]
                x_uncertainty_per_patch = x_uncertainty_per_patch[:, :, :-1]

            x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),obs_num_per_patch)

            # x, _ = torch.max(x, dim=3)

            x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            x_uncertainty = x_uncertainty_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch).squeeze(-1)
            if M == 1:
                return torch.squeeze(x)

            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = (obs_num_per_patch > 0).float().view(B, N, M // 2, 2, 1)
            x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)
        # obs_num_per_patch[torch.where(obs_num_per_patch == 0)] = obs_num_per_patch[torch.where(obs_num_per_patch == 0)] + 1
        # x[torch.isinf(x)] = 0
        # x[torch.isnan(x)] = 0
        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):

        """
        time_steps_to_predict (B, L) [0, 1]
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>

        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.nodevec_dim).repeat(B, 1, M, L_in, 1)
        X = torch.cat([X, te_his, var_emb], dim=-1)  # (B*N*M, L, F)
        # print(X.shape, te_his.shape)

        non_zero_ind = torch.where(mask.reshape(B, N, -1))
        non_zero_tp = truth_time_steps.reshape(B, N, -1)[non_zero_ind]

        x_uncertainty = torch.zeros_like(mask.reshape(B, N, -1))


        L = len(non_zero_tp)

        time_steps_padded = torch.nn.functional.pad(non_zero_tp, (1, 1), value=0)
        prev_intervals = time_steps_padded[1:L + 1] - time_steps_padded[:L]
        next_intervals = time_steps_padded[2:L + 2] - time_steps_padded[1:L + 1]

        prev_intervals[torch.where(prev_intervals <= 0)] = prev_intervals[torch.where(prev_intervals <= 0)] + 2
        next_intervals[torch.where(next_intervals <= 0)] = next_intervals[torch.where(next_intervals <= 0)] + 2

        uncertainty = torch.minimum(prev_intervals, next_intervals)

        uncertainty[torch.where(uncertainty > 1)] = torch.max(uncertainty[torch.where(uncertainty < 1)])

        x_uncertainty[non_zero_ind] = uncertainty


        x_uncertainty = x_uncertainty.view(B, N, M, L_in)
        x_uncertainty = torch.tanh(x_uncertainty) * mask.squeeze()
        # x_uncertainty = (x_uncertainty - torch.min(x_uncertainty)) / (torch.max(x_uncertainty) - torch.min(x_uncertainty))

        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, x_uncertainty)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)
        # print(outputs.shape)
        # assert not torch.any(torch.isnan(outputs))

        return outputs  # (1, B, Lp, N)


class HPG_WOGraph(nn.Module):
    def __init__(self, args, supports=None, dropout=0):

        super(HPG_WOGraph, self).__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer

        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim - 1)

        self.obs_enc = nn.Linear(1, args.hid_dim)

        d_model = args.hid_dim

        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)

        ### Encoder output layer ###
        self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        ### Decoder ###
        self.decoder = nn.Sequential(
            nn.Linear(nodevec_dim + enc_dim + 2 * args.te_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1
        for i in range(layer_nums):
            B, N, M, L, D = x.shape
            # 池化聚合同一Patch 同一变量的隐藏状态
            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]

            if M > 1 and M % 2 != 0:
                node_state_sum_per_patch[:, :, -2, :] += node_state_sum_per_patch[:, :, -1, :]
                node_state_sum_per_patch = node_state_sum_per_patch[:, :, :-1, :]

                obs_num_per_patch[:, :, -2, :] += obs_num_per_patch[:, :, -1, :]
                obs_num_per_patch = obs_num_per_patch[:, :, :-1, :]

                x_time_per_patch[:, :, -2, :] += x_time_per_patch[:, :, -1, :]
                x_time_per_patch = x_time_per_patch[:, :, :-1, :]

            x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                       obs_num_per_patch)
            x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            if M == 1:
                return torch.squeeze(x)

            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = (obs_num_per_patch > 0).float().view(B, N, M // 2, 2, 1)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):

        """
        time_steps_to_predict (B, L) [0, 1]
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>

        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.nodevec_dim).repeat(B, 1, M, L_in, 1)
        X = torch.cat([X, te_his, var_emb], dim=-1)  # (B*N*M, L, F)
        # print(X.shape, te_his.shape)

        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)
        # print(outputs.shape)
        # assert not torch.any(torch.isnan(outputs))

        return outputs  # (1, B, Lp, N)


class GTrans(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, dropout=0.1, **kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_k // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)

        # Attention Layer Initialization
        self.w_k_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for i in range(self.n_heads)])
        self.w_v_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for i in range(self.n_heads)])
        self.w_v_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])

        # self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_transfer = nn.ModuleList([nn.Linear(self.d_input + 1, self.d_k, bias=True) for i in range(self.n_heads)])

        # initiallization
        utils.init_network_weights(self.w_k_list_same)
        utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list_same)
        utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)

        # Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        # Normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value, time_nodes, edge_same):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value, edge_same=edge_same, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        edge_same = edge_same.view(-1, 1)
        for i in range(self.n_heads):
            k_linear_same = self.w_k_list_same[i]
            k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear_same = self.w_v_list_same[i]
            v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1, 1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer, k_linear_same, k_linear_diff, q_linear, x_i,
                                                 edge_same)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            # attention_norm = softmax(attention, edge_index_i)  # [4,1]
            # attention_norm = torch.softmax(attention)  # [4,1]
            attention_norm = attention  # [4,1]
            # attention_norm = (attention - torch.min(attention)) / (torch.max(attention) - torch.min(attention))

            sender_same = edge_same * v_linear_same(x_j_transfer)
            sender_diff = (1 - edge_same) * v_linear_diff(x_j_transfer)
            sender = sender_same + sender_diff

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k_same, w_k_diff, w_q, x_i, edge_same):
        x_i = w_q(x_i)  # receiver #[num_edge,d*heads]

        # wraping k

        sender_same = edge_same * w_k_same(x_j_transfer)
        sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class HPG(nn.Module):
    def __init__(self, args, supports=None, dropout=0):

        super(HPG, self).__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()

        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim - 1)

        self.obs_enc = nn.Linear(1, args.hid_dim)

        d_model = args.hid_dim



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)


        ### Encoder output layer ###
        self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            self.gcs.append(GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, dropout))

        ### Decoder ###
        self.decoder = nn.Sequential(
            # nn.Linear(enc_dim+args.te_dim, args.hid_dim),
            nn.Linear(nodevec_dim + enc_dim + 2 * args.te_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )



    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        for i in range(layer_nums):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            # print(cur_adj.shape)

            edge_ind = torch.where(cur_adj == 1)
            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            # edge_mask = torch.where(edge_time <= 0)[0]
            # edge_index = edge_index[:, edge_mask]
            # edge_time = edge_time[edge_mask]

            # edge_same = (edge_time < 100).float()
            edge_same = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_same = torch.ones_like(edge_time).to(edge_time.device)

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]

            if M > 1 and M % 2 != 0:
                node_state_sum_per_patch[:, :, -2, :] += node_state_sum_per_patch[:, :, -1, :]
                node_state_sum_per_patch = node_state_sum_per_patch[:, :, :-1, :]

                # x[:, :, -2] = torch.max(x[:, :, -2], x[:, :, -1])
                # x = x[:, :, :-1]

                obs_num_per_patch[:, :, -2, :] += obs_num_per_patch[:, :, -1, :]
                obs_num_per_patch = obs_num_per_patch[:, :, :-1, :]

                x_time_per_patch[:, :, -2, :] += x_time_per_patch[:, :, -1, :]
                x_time_per_patch = x_time_per_patch[:, :, :-1, :]

            x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),obs_num_per_patch)

            # x, _ = torch.max(x, dim=3)

            x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            if M == 1:
                return torch.squeeze(x)

            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = (obs_num_per_patch > 0).float().view(B, N, M // 2, 2, 1)
        # obs_num_per_patch[torch.where(obs_num_per_patch == 0)] = obs_num_per_patch[torch.where(obs_num_per_patch == 0)] + 1
        # x[torch.isinf(x)] = 0
        # x[torch.isnan(x)] = 0
        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):

        """
        time_steps_to_predict (B, L) [0, 1]
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>

        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.nodevec_dim).repeat(B, 1, M, L_in, 1)
        X = torch.cat([X, te_his, var_emb], dim=-1)  # (B*N*M, L, F)
        # print(X.shape, te_his.shape)

        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)
        # print(outputs.shape)
        # assert not torch.any(torch.isnan(outputs))

        return outputs  # (1, B, Lp, N)




class BaselineHPG_Fixed(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_Fixed, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)



class BaselineHPG_Fixed_WOWJTransfer(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_Fixed_WOWJTransfer, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans_WOWJTransfer(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)


class BaselineGTrans_WJTransferWOTE(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(BaselineGTrans_WJTransferWOTE, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_k // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        # self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=True) for i in range(self.n_heads)])

        self.w_transfer = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, self.d_input, self.d_input)) for i in range(self.n_heads)])
        for param in self.w_transfer:
            nn.init.xavier_uniform_(param)

        # initiallization
        # init_network_weights(self.w_k_list)
        # init_network_weights(self.w_q_list)
        # init_network_weights(self.w_v_list)
        # init_network_weights(self.w_transfer)
        # for param in self.w_transfer:
        #     nn.init.xavier_uniform_(param)

        # Temporal Layer
        # self.temporal_net = TemporalEncoding(d_input)

        # self.te_scale = nn.Linear(1, 1)
        # self.te_periodic = nn.Linear(1, d_input - 1)

        self.layer_norm = nn.LayerNorm(d_input)

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]

            # v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i][n_layer]
            #
            # edge_temporal_true = self.temporal_net(edges_temporal)
            # edge_temporal_true = self.LearnableTE(edges_temporal.unsqueeze(-1))
            # edges_temporal = edges_temporal.view(-1, 1)
            # x_j_transfer = F.gelu(w_transfer(torch.cat([x_j, edges_temporal], dim=1))) + edge_temporal_true
            x_j_transfer = F.gelu(torch.matmul(x_j, w_transfer))
            # x_j_transfer = x_j
            # x_j_transfer = F.gelu(torch.matmul(torch.cat([x_j, edges_temporal], dim=1), w_transfer))
            # x_j_transfer = F.gelu(x_j)

            # x_j_transfer = F.gelu(w_transfer(torch.cat([x_j, edges_temporal], dim=1)))
            # x_j_transfer = x_j
            # x_j_transfer = F.gelu(w_transfer(torch.cat([x_j, edges_temporal], dim=1)))

            # x_j_transfer = F.gelu(w_transfer(x_j))

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1]) # receiver #[num_edge,d*heads]
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1 + x_i_2
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class BaselineHPG_Fixed_WJTransferWOTE(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_Fixed_WJTransferWOTE, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans_WJTransferWOTE(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineHPG_Fixed_AGGAddTE(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_Fixed_AGGAddTE, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            # x = torch.sum((V * scale_attention), dim=-2)
            x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineHPG_Fixed_GatedFusion(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_Fixed_GatedFusion, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.output_gate_w = nn.Parameter(torch.FloatTensor(args.patch_layer, d_model, 1))
        self.output_gate_b = nn.Parameter(torch.FloatTensor(args.patch_layer, 1))
        nn.init.xavier_uniform_(self.output_gate_w)
        nn.init.xavier_uniform_(self.output_gate_b)

        self.output_w = nn.Parameter(torch.FloatTensor(args.patch_layer, d_model, d_model))
        self.output_b = nn.Parameter(torch.FloatTensor(args.patch_layer, d_model))
        nn.init.xavier_uniform_(self.output_w)
        nn.init.xavier_uniform_(self.output_b)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        output = torch.zeros(B, N, D).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            gate_x = torch.matmul(x, self.output_gate_w[n_layer]) + self.output_gate_b[n_layer]
            gate_x[torch.where(mask_X == 0)] = -1e10
            gate_x = torch.softmax(gate_x, dim=-2)
            out_x = torch.matmul(x, self.output_w[n_layer]) + self.output_b[n_layer]
            out_x = out_x * gate_x
            output = output + torch.sum(out_x, dim=-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                gate_x = torch.matmul(x, self.output_gate_w[n_layer]) + self.output_gate_b[n_layer]
                gate_x[torch.where(mask_X == 0)] = -1e10
                gate_x = torch.softmax(gate_x, dim=-2)
                out_x = torch.matmul(x, self.output_w[n_layer]) + self.output_b[n_layer]
                out_x = out_x * gate_x
                output = output + torch.sum(out_x, dim=-2)
                return output
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineHPG_LSTEQKV(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_LSTEQKV, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)

        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # self.w_q = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))
        # self.w_k = nn.Parameter(torch.FloatTensor(args.te_dim, nodevec_dim + enc_dim + args.te_dim))

        self.w_q = nn.Parameter(torch.FloatTensor(args.patch_layer,d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(args.patch_layer,d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(args.patch_layer,d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)



        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
            # cur_x_uncertainty = rearrange(x_uncertainty, 'b n m l -> (b m n l)')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))

            int_max = torch.iinfo(torch.int32).max
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B // once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed].unsqueeze(0) == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q[n_layer])
            K = torch.matmul(time_te, self.w_k[n_layer])
            V = torch.matmul(x, self.w_v[n_layer])
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            # x_time_emb = self.LearnableTE(x_time)
            # x = torch.cat([x, x_time_emb], dim=-1)
            # x_time = torch.sum((x_time.reshape(-1, L, 1) * scale_attention.unsqueeze(-1)), dim=1).reshape(B, N, M, 1)




            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = X + var_emb + te_his  # (B*N*M, L, F)


        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)
        # h = torch.mean(h, dim=-1)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class BaselineGTrans(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(BaselineGTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        self.layer_norm = nn.LayerNorm(d_input)

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]


            x_j_transfer = x_j

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1]) # receiver #[num_edge,d*heads]
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1 + x_i_2
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class BaselineHPG(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = torch.matmul(x+avg_te, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineHPG_v2(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_v2, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        B, N, M, L, D = x.shape

        # 将其扩展成形状为 [1, N, 1, 1, 1]
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

        # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
        # 并行式
        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        # 生成图结构
        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
        int_max = torch.iinfo(torch.int32).max
        element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

        if element_count > int_max:
            once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
            sd = 0
            ed = once_num
            total_num = math.ceil(B / once_num)
            for k in range(total_num):
                if k == 0:
                    edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = edge_ind[0]
                    edge_ind_1 = edge_ind[1]
                    edge_ind_2 = edge_ind[2]
                    edge_ind_3 = edge_ind[3]
                elif k == total_num - 1:
                    cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                else:
                    cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                sd += once_num
                ed += once_num

        else:
            edge_ind = torch.where(cur_adj == 1)

        source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
        target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0

        # 图神经网络传播节点状态
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

        # 池化聚合同一Patch 同一变量的隐藏状态
        # 若Patch为奇数个，创建一个虚拟节点
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
        x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
        time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)
        x_time = avg_x_time

        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape

            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)

            cur_variable_indices = variable_indices.view(1, N, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, T, 1).reshape(-1, 1)

            patch_indices = torch.arange(T).float().to(x.device)

            cur_patch_indices = patch_indices.view(1, 1, T)
            missing_indices = torch.where(mask_X.reshape(B, -1) == 0)

            cur_patch_indices = cur_patch_indices.expand(B, N, T).reshape(B, -1)

            patch_indices_matrix_1 = cur_patch_indices.unsqueeze(1).expand(B, N * T, N * T)
            patch_indices_matrix_2 = cur_patch_indices.unsqueeze(-1).expand(B, N * T, N * T)

            patch_interval = patch_indices_matrix_1 - patch_indices_matrix_2
            patch_interval[missing_indices[0], missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)
            patch_interval[missing_indices[0], :, missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)
            # cur_adj = patch_interval == 1 and patch_interval == -1
            if len(patch_interval[0] > 0):
                edge_ind = torch.where(torch.abs(patch_interval) == 1)

                source_nodes = (N * T * edge_ind[0] + edge_ind[1])
                target_nodes = (N * T * edge_ind[0] + edge_ind[2])
                edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

                edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

                edge_diff_time_same_var = (
                            (cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
                edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
                edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
                edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
                edge_same_time_diff_var[edge_self] = 0.0

                # 图神经网络传播节点状态
                for gc in self.gcs:
                    cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var,
                               edge_diff_time_diff_var, n_layer)
                x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1

            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                        obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)

            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            x_time = avg_x_time

            # x = x.view(B, N, M // 2, 2, D)
            # x_time = x_time.view(B, N, M // 2, 2, 1)
            # mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)



        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineHPG_v2_1Layer(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_v2_1Layer, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim


        self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, 1, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        B, N, M, L, D = x.shape

        # 将其扩展成形状为 [1, N, 1, 1, 1]
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

        # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
        # 并行式
        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        # 生成图结构
        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
        int_max = torch.iinfo(torch.int32).max
        element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

        if element_count > int_max:
            once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
            sd = 0
            ed = once_num
            total_num = math.ceil(B / once_num)
            for k in range(total_num):
                if k == 0:
                    edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = edge_ind[0]
                    edge_ind_1 = edge_ind[1]
                    edge_ind_2 = edge_ind[2]
                    edge_ind_3 = edge_ind[3]
                elif k == total_num - 1:
                    cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                else:
                    cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                sd += once_num
                ed += once_num

        else:
            edge_ind = torch.where(cur_adj == 1)

        source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
        target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0

        # 图神经网络传播节点状态
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)
        x = x.reshape(B, N, -1, D).unsqueeze(2)
        mask_X = mask_X.reshape(B, N, -1, 1).unsqueeze(2)
        x_time = x_time.reshape(B, N, -1, 1).unsqueeze(2)

        obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
        x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
        time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
        x = torch.sum((V * scale_attention), dim=-2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)


class BaselineHPG_v2_AddLayer1(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_v2_AddLayer1, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.scale_patch_size = args.scale_patch_size
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim
        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()

        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        B, N, M, L, D = x.shape

        # 将其扩展成形状为 [1, N, 1, 1, 1]
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

        # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)

        # 并行式
        cur_x = rearrange(x, 'b n m l c -> (b n m l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b n m l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b n m l) c')


        cur_t_indices = rearrange(x_time, 'b n m l c -> b (n m l) c')

        missing_indices = torch.where(mask_X.reshape(B, -1) == 0)

        cur_t_indices = cur_t_indices.reshape(B, -1)

        t_indices_matrix_1 = cur_t_indices.unsqueeze(1).expand(B, N * M * L, N * M * L)
        t_indices_matrix_2 = cur_t_indices.unsqueeze(-1).expand(B, N * M * L, N * M * L)

        t_interval = t_indices_matrix_1 - t_indices_matrix_2
        t_interval[missing_indices[0], missing_indices[1]] = torch.ones(len(missing_indices[0]), N * M * L).to(
            x.device)
        t_interval[missing_indices[0], :, missing_indices[1]] = torch.ones(len(missing_indices[0]), N * M * L).to(
            x.device)
        # cur_adj = patch_interval == 1 and patch_interval == -1

        edge_ind = torch.where(torch.abs(t_interval) < self.scale_patch_size)

        source_nodes = (N * M * L * edge_ind[0] + edge_ind[1])
        target_nodes = (N * M * L * edge_ind[0] + edge_ind[2])

        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0

        # 图神经网络传播节点状态
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b n m l) c -> b n m l c', b=B, n=N, m=M, l=L)

        # 池化聚合同一Patch 同一变量的隐藏状态
        # 若Patch为奇数个，创建一个虚拟节点
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
        x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
        time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)
        x_time = avg_x_time

        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape

            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)

            cur_variable_indices = variable_indices.view(1, N, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, T, 1).reshape(-1, 1)

            patch_indices = torch.arange(T).float().to(x.device)

            cur_patch_indices = patch_indices.view(1, 1, T)
            missing_indices = torch.where(mask_X.reshape(B, -1) == 0)

            cur_patch_indices = cur_patch_indices.expand(B, N, T).reshape(B, -1)

            patch_indices_matrix_1 = cur_patch_indices.unsqueeze(1).expand(B, N * T, N * T)
            patch_indices_matrix_2 = cur_patch_indices.unsqueeze(-1).expand(B, N * T, N * T)

            patch_interval = patch_indices_matrix_1 - patch_indices_matrix_2
            patch_interval[missing_indices[0], missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)
            patch_interval[missing_indices[0], :, missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)
            # cur_adj = patch_interval == 1 and patch_interval == -1

            edge_ind = torch.where(torch.abs(patch_interval) == 1)

            source_nodes = (N * T * edge_ind[0] + edge_ind[1])
            target_nodes = (N * T * edge_ind[0] + edge_ind[2])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = (
                        (cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var,
                           edge_diff_time_diff_var, n_layer)
            x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1

            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                        obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)

            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            x_time = avg_x_time

            # x = x.view(B, N, M // 2, 2, D)
            # x_time = x_time.view(B, N, M // 2, 2, 1)
            # mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)



        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)


class BaselineHPG_LastRef(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_LastRef, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1


            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)
            last_time = torch.fill_(torch.zeros(B, N, 1, 1), torch.max(x_time)).to(x.device)

            last_time_te = self.LearnableTE(last_time)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)

            Q = torch.matmul(last_time_te, self.w_q)
            K = torch.matmul(time_te.reshape(B, N, -1, D), self.w_k)
            V = torch.matmul(x.reshape(B, N, -1, D), self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X.reshape(B, N, -1, 1) == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            if n_layer == 0:
                cur_scale_rep = torch.sum((V * scale_attention), dim=-2)
            else:
                cur_scale_rep = cur_scale_rep + torch.sum((V * scale_attention), dim=-2)

            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)

            x_time = avg_x_time
            if M == 1:
                return cur_scale_rep
                # return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)


class BaselineHPG_WOTEAGG(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_WOTEAGG, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)
            # avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            # time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            # Q = torch.matmul(avg_te, self.w_q)
            # K = torch.matmul(time_te, self.w_k)
            # V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            # attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            # attention = torch.div(attention, Q.shape[-1] ** 0.5)
            # attention[torch.where(mask_X == 0)] = -1e10
            # scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            # x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineGTrans_WOLayerSpecParam(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(BaselineGTrans_WOLayerSpecParam, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(1, 3, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        self.layer_norm = nn.LayerNorm(d_input)

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        n_layer = 0
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]


            x_j_transfer = x_j

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1]) # receiver #[num_edge,d*heads]
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1 + x_i_2
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class BaselineHPG_WOLayerSpecParam(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_WOLayerSpecParam, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans_WOLayerSpecParam(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineGTrans_WO3E(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(BaselineGTrans_WO3E, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 1, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        self.layer_norm = nn.LayerNorm(d_input)

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]


            x_j_transfer = x_j

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1 + x_i_2
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender = sender_0 + sender_1 + sender_2
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class BaselineHPG_WO3E(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_WO3E, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans_WO3E(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)

class BaselineGTrans_WODTDV(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(BaselineGTrans_WODTDV, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 2, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        self.layer_norm = nn.LayerNorm(d_input)

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]


            x_j_transfer = x_j

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[1]) + bias_v[1])
            # sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1]) # receiver #[num_edge,d*heads]
        # x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        # sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class BaselineHPG_WODTDV(nn.Module):
    def __init__(self, args, supports=None):
        super(BaselineHPG_WODTDV, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        # self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)



        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim

        self.nodevec = nn.Parameter(torch.randn(self.N, d_model).cuda(), requires_grad=True)
        self.relu = nn.ReLU()
        # self.nodevec = nn.Parameter(torch.FloatTensor(self.N, d_model))
        # nn.init.xavier_uniform_(self.nodevec)


        ### Encoder output layer ###
        # self.outlayer = args.outlayer
        enc_dim = args.hid_dim

        for l in range(self.n_layer):
            # self.gcs.append(UA_GTrans(1, nodevec_dim + enc_dim + args.te_dim, nodevec_dim + enc_dim + args.te_dim, self.alpha))
            self.gcs.append(BaselineGTrans_WODTDV(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        """
        x (B, N, M, L, F)
        mask_X (B, N, M, L, 1)
        x_time (B, N, M, L, 1)
        """
        B, N, M, L, D = x.shape
        layer_nums = int(math.log2(M)) + 1

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        # for i in range(layer_nums + 1):
        for n_layer in range(self.patch_layer):
            B, N, M, L, D = x.shape

            # 将其扩展成形状为 [1, N, 1, 1, 1]
            cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
            # 并行式
            cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
            cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
            cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

            # 生成图结构
            cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
            cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
            # cur_adj_0 = torch.matmul(cur_mask[:B // 3], cur_mask[:B // 3].permute(0, 1, 3, 2))
            # cur_adj_1 = torch.matmul(cur_mask[B//3:2*B//3], cur_mask[B//3:2*B//3].permute(0, 1, 3, 2))
            # cur_adj_2 = torch.matmul(cur_mask[2*B//3:], cur_mask[2*B//3:].permute(0, 1, 3, 2))
            # cur_adj = torch.cat([cur_adj_0, cur_adj_1, cur_adj_2], dim=0)

            int_max = torch.iinfo(torch.int32).max
            # int_max = 1100000000
            # int_max = 32550000
            element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

            if element_count > int_max:
                once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
                sd = 0
                ed = once_num
                total_num = math.ceil(B / once_num)
                for k in range(total_num):
                    if k == 0:
                        edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = edge_ind[0]
                        edge_ind_1 = edge_ind[1]
                        edge_ind_2 = edge_ind[2]
                        edge_ind_3 = edge_ind[3]
                    elif k == total_num - 1:
                        cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                        edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                    else:
                        cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                        edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                        edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                        edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                        edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    sd += once_num
                    ed += once_num

            else:
                edge_ind = torch.where(cur_adj == 1)

            source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
            target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            # edge_same_time_diff_var= ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            # edge_diff_time_same_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()

            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            # edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) > 100).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
            # edge_diff_time_same_var[edge_self] = 0.0

            # 图神经网络传播节点状态
            # cur_x = self.gcs(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            for gc in self.gcs:
                cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
            # cur_x = self.base_conv(cur_x, edge_index, edge_time, cur_x_time, edge_same)
            # x = cur_x.reshape(B, N, M, L, D)
            x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if M > 1 and M % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
                M = M + 1

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                    obs_num_per_patch)

            # avg_x_time = torch.zeros_like(obs_num_per_patch).to(x.device)
            # avg_x_time[torch.where(obs_num_per_patch == 1)] = torch.max(x_time, dim=3)[0][torch.where(obs_num_per_patch == 1)]
            # avg_x_time[torch.where(obs_num_per_patch > 1)] = ((torch.max(x_time, dim=3)[0] + torch.min(x_time, dim=3)[0]) / 2)[torch.where(obs_num_per_patch > 1)]
            # node_state_sum_per_patch = torch.sum(x, dim=3)  # x.shape[B, N, M, L, D]


            # x = node_state_sum_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
            #                                         obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            # V = x


            # K = torch.matmul(x, self.w_k).view(-1 ,L, D)
            # Q = F.normalize(Q, p=2, dim=-1)
            # K = F.normalize(K, p=2, dim=-1)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            # attention = torch.matmul(Q.unsqueeze(-2), K.unsqueeze(-1))
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            # x = torch.sum((V * scale_attention), dim=-2) + avg_te.squeeze(-2)

            x_time = avg_x_time
            if M == 1:
                return torch.squeeze(x)


            x = x.view(B, N, M // 2, 2, D)
            x_time = x_time.view(B, N, M // 2, 2, 1)
            mask_X = mask_X.view(B, N, M // 2, 2, 1)
            # x_uncertainty = x_uncertainty.view(B, N, M // 2, 2)

        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
    # def forward(self, observed_tp, observed_data, observed_mask, tau, return_almat=False):
        """
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>
        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs  # (1, B, Lp, N)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        # x (B, F, N, M)
        # A (B, M, N, N)
        x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
        # print(x.shape)
        return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear,self).__init__()
        # self.mlp = nn.Linear(c_in, c_out)
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        # x (B, F, N, M)

        # return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        # c_in = (order*support_len)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x (B, F, N, M)
        # a (B, M, N, N)
        out = [x]
        for a in support:
            # print(x.shape, a.shape)
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1) # concat x and x_conv
        h = self.mlp(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe[:, :x.size(1), :].shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1), :]
        return x





class tPatchGNN(nn.Module):
    def __init__(self, args, supports=None, dropout=0):

        super(tPatchGNN, self).__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer

        ### Intra-time series modeling ##
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim - 1)

        ## TTCN
        input_dim = 1 + args.te_dim
        ttcn_dim = args.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim, bias=True))
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
        # nn.init.normal_(self.T_bias.data)
        # nn.init.kaiming_normal_(self.T_bias.data, mode='fan_out', nonlinearity='relu')
        # self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        d_model = args.hid_dim
        ## Transformer
        self.ADD_PE = PositionalEncoding(d_model)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=args.tf_layer))
        # self.transformer_encoder = AttentionLayer(FullAttention(), d_model, args.nhead)

        ### Inter-time series modeling ###
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim
        if supports is None:
            self.supports = []

        self.nodevec1 = nn.Parameter(torch.randn(self.N, nodevec_dim).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.N).cuda(), requires_grad=True)
        # nn.init.normal_(self.nodevec1.data)
        # nn.init.normal_(self.nodevec2.data)
        # nn.init.kaiming_normal_(self.nodevec1.data, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.nodevec2.data, mode='fan_out', nonlinearity='relu')

        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.n_layer):
            self.nodevec_linear1.append(nn.Linear(args.hid_dim, nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(args.hid_dim, nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
                nn.Linear(args.hid_dim + nodevec_dim, 1),
                # nn.Linear(args.hid_dim+nodevec_dim, nodevec_dim),
                nn.Tanh(),
                nn.ReLU()))
            self.nodevec_gate2.append(nn.Sequential(
                nn.Linear(args.hid_dim + nodevec_dim, 1),
                # nn.Linear(args.hid_dim+nodevec_dim, nodevec_dim),
                nn.Tanh(),
                nn.ReLU()))

        self.supports_len += 1

        self.gconv = nn.ModuleList()  # gragh conv
        for _ in range(self.n_layer):
            self.gconv.append(gcn(d_model, d_model, dropout, support_len=self.supports_len, order=args.hop))

        # self.bn = nn.ModuleList()
        # for _ in range(self.n_layer):
        # 	self.bn.append(nn.BatchNorm2d(args.hid_dim))

        ### Encoder output layer ###
        self.outlayer = args.outlayer
        enc_dim = args.hid_dim
        if (self.outlayer == "Linear"):
            self.temporal_agg = nn.Sequential(
                nn.Linear(args.hid_dim * self.M, enc_dim))

        elif (self.outlayer == "CNN"):
            self.temporal_agg = nn.Sequential(
                nn.Conv1d(d_model, enc_dim, kernel_size=self.M))

        ### Decoder ###
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim + args.te_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1)
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def TTCN(self, X_int, mask_X):
        # X_int: shape (B*N*M, L, F)
        # mask_X: shape (B*N*M, L, 1)

        N, Lx, _ = mask_X.shape
        # assert not torch.any(torch.isnan(X_int))
        Filter = self.Filter_Generators(X_int)  # (N, Lx, F_in*ttcn_dim)
        # assert not torch.any(torch.isnan(Filter))
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1)  # (N, Lx, ttcn_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1)  # (N, ttcn_dim)
        # print(mask_X.shape, Filter_seqnorm.shape, ttcn_out.shape)
        h_t = torch.relu(ttcn_out + self.T_bias)  # (N, ttcn_dim)
        # h_t = self.leakyrelu(ttcn_out + self.T_bias) # (N, ttcn_dim)
        # h_t = torch.tanh(ttcn_out + self.T_bias) # (N, ttcn_dim)
        return h_t

    def IMTS_Model(self, x, mask_X):
        """
        x (B*N*M, L, F)
        mask_X (B*N*M, L, 1)
        """
        # print(mask_X[...,0])
        # mask for the patch
        mask_patch = (mask_X.sum(dim=1) > 0)  # (B*N*M, 1)

        ### TTCN for patch modeling ###
        x_patch = self.TTCN(x, mask_X)  # (B*N*M, hid_dim-1)
        # assert not torch.any(torch.isnan(x))
        x_patch = torch.cat([x_patch, mask_patch], dim=-1)  # (B*N*M, hid_dim)
        x_patch = x_patch.view(self.batch_size, self.N, self.M, -1)  # (B, N, M, hid_dim)
        B, N, M, D = x_patch.shape

        x = x_patch
        for layer in range(self.n_layer):

            if (layer > 0):  # residual
                x_last = x.clone()

            ### Transformer for temporal modeling ###
            x = x.reshape(B * N, M, -1)  # (B*N, M, F)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x).view(x_patch.shape)  # (B, N, M, F)

            ### unidirectional mask
            # attn_mask = (torch.tril(torch.ones(M, M)) == 0).to(self.device)
            # x_tf = self.transformer_encoder(x_tf, mask=attn_mask).view(x_patch.shape) # (B, N, M, F)
            # assert not torch.any(torch.isnan(x))

            ### GNN for inter-time series modeling ###
            ### gated adaptive graph learning ###
            nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)
            # print(x.shape, nodevec1.shape, nodevec2.shape)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            x_p1 = x_gate1 * self.nodevec_linear1[layer](x)  # (B, M, N, 10)
            x_p2 = x_gate2 * self.nodevec_linear2[layer](x)  # (B, M, N, 10)
            nodevec1 = nodevec1 + x_p1.permute(0, 2, 1, 3)  # (B, M, N, 10)
            nodevec2 = nodevec2 + x_p2.permute(0, 2, 3, 1)  # (B, M, 10, N)

            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1)  # (B, M, N, N) used
            # adp = F.softmax(torch.matmul(nodevec1, nodevec2), dim=-1) # (B, M, N, N) # try this one with ct gcn in ushcn
            # adp = F.relu(F.tanh(torch.matmul(d_nodevec1, d_nodevec2))) # (B, M, N, N)
            # adp = adp / (adp.sum(dim=-2, keepdim=True)+1e-8)
            # adp = F.sigmoid(torch.matmul(d_nodevec1, d_nodevec2)) # (B, M, N, N)
            # print(d_nodevec1.shape, d_nodevec2.shape, adp.shape)
            new_supports = self.supports + [adp]

            # input x shape (B, F, N, M)
            x = self.gconv[layer](x.permute(0, 3, 1, 2), new_supports)  # (B, F, N, M)
            # assert not torch.any(torch.isnan(x))
            x = x.permute(0, 2, 3, 1)  # (B, N, M, F)

            if (layer > 0):  # residual addition
                x = x_last + x

        # x = x.permute(0, 3, 1, 2) # (B, F, N, M)
        # x = self.bn[layer](x)
        # x = x.permute(0, 2, 3, 1) # (B, N, M, F)

        ### Output layer ###
        if (self.outlayer == "CNN"):
            x = x.reshape(self.batch_size * self.N, self.M, -1).permute(0, 2, 1)  # (B*N, F, M)
            x = self.temporal_agg(x)  # (B*N, F, M) -> (B*N, F, 1)
            x = x.view(self.batch_size, self.N, -1)  # (B, N, F)

        elif (self.outlayer == "Linear"):
            x = x.reshape(self.batch_size, self.N, -1)  # (B, N, M*F)
            x = self.temporal_agg(x)  # (B, N, hid_dim)

        return x

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):

        """
        time_steps_to_predict (B, L) [0, 1]
        X (B, M, L, N)
        truth_time_steps (B, M, L, N) [0, 1]
        mask (B, M, L, N)

        To ====>

        X (B*N*M, L, 1)
        truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

        # print("data shape:", time_steps_to_predict.shape, X.shape, truth_time_steps.shape, mask.shape)
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B*N*M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)

        X = torch.cat([X, te_his], dim=-1)  # (B*N*M, L, F)
        # print(X.shape, te_his.shape)

        ### *** a encoder to model irregular time series
        # assert not torch.any(torch.isnan(X))
        h = self.IMTS_Model(X, mask)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        h = torch.cat([h, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)
        # print(outputs.shape)
        # assert not torch.any(torch.isnan(outputs))

        return outputs  # (1, B, Lp, N)


