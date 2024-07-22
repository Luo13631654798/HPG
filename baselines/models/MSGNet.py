import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from ..layers.Embed import DataEmbedding
from ..layers.MSGBlock import GraphBlock, Attention_Block


def FFT_for_Period(x, k=2):
    # [B, T, C]
    # B是批次大小，T是时间步长，C是特定维度
    xf = torch.fft.rfft(x, dim=1) # 傅里叶变换，得到[B,T/2+1,C]
    frequency_list = abs(xf).mean(0).mean(-1)# 计算傅里叶变换幅度的均值，[T/2+1]
    frequency_list[0] = 0 # 将频率列表第一个元素置为0
    _, top_list = torch.topk(frequency_list, k) # 获取频率列表前k个最大值索引
    top_list = top_list.detach().cpu().numpy() # 将索引从张量转为numpy
    period = x.shape[1] // top_list # 计算周期
    return period, abs(xf).mean(-1)[:, top_list] # 返回周期何对应频率的幅度


class ScaleGraphBlock(nn.Module):  # ScaleGraphBlock
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                    n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out, configs.d_model, configs.conv_channel, configs.skip_channel,
                           configs.gcn_depth, configs.dropout, configs.propalpha, configs.seq_len,
                           configs.node_dim))

    def forward(self, x):
        B, T, N = x.size() # B：批次大小，T：时间步长，N是节点数
        scale_list, scale_weight = FFT_for_Period(x, self.k) # 计算周期何频率幅度
        res = [] # 存储每个尺度的结果
        for i in range(self.k):
            scale = scale_list[i]
            # Gconv
            x = self.gconv[i](x)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

            # for Mul-attetion
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)
            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ### add linear ##########
        # self.linear_layer = nn.Linear(12, self.configs.enc_in)

        # 编码器encoder
        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           'timeF', 'a', configs.dropout)
        self.layer = configs.e_layers # 层数
        self.layer_norm = nn.LayerNorm(configs.d_model) # 层归一化
        self.projection = nn.Linear(1,configs.enc_in)
        # 解码器decoder
        self.te_scale = nn.Linear(1, 1)  # 线性层，用于时间嵌入的缩放
        self.te_periodic = nn.Linear(1, configs.d_model - 1)  # 线性层，用于时间嵌入的周期性部分

        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),  # 线性层
            nn.ReLU(inplace=True),  # relu激活函数
            nn.Linear(configs.d_model, configs.d_model),  # 线性层
            nn.ReLU(inplace=True),  # relu激活函数
            nn.Linear(configs.d_model, 1)  # 线性层，输出为1
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)，输入时间序列
        out1 = self.te_scale(tt)  # 时间嵌入的缩放
        out2 = torch.sin(self.te_periodic(tt))  # 时间嵌入的周期性部分
        return torch.cat([out1, out2], -1)  # 将缩放和周期性部分拼接

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        # Normalization from Non-stationary Transformer
        '''
        means = observed_data.mean(1, keepdim=True).detach()
        x_enc = observed_data - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev # 进行标准化
        '''
        x_enc = observed_data

        # 调整通道数以匹配模型的期望输入
        # x_enc = self.linear_layer(x_enc) # 线性变换，输出形状[B,T,enc_in]
        _, _, N = x_enc.shape
        ### add the content ###
        padding_len = self.seq_len - x_enc.shape[1] # 计算填充长度
        padding = torch.zeros(size=[x_enc.shape[0], padding_len, x_enc.shape[2]]).to(observed_data.device)
        # 创建填充张量
        x_enc = torch.cat([x_enc, padding], dim=1)
        padding_t = torch.zeros(size=[observed_tp.shape[0], padding_len]).to(observed_data.device)
        # 创建填充时间序列
        observed_tp = torch.cat([observed_tp, padding_t], dim=1)

        # print("Shape of observed_tp: ", observed_tp.shape) # [32 96]
        # 确保 observed_tp 的形状为 [batch_size, seq_len, 1]
        # observed_tp = observed_tp.unsqueeze(-1) 
        # print("Shape of observed_tp_2: ", observed_tp.shape) # [32 96 1]
        # print("Shape of x_enc after embedding:", x_enc.shape) # [32 96 7]

        # embedding
        enc_out = self.enc_embedding(x_enc, observed_tp.unsqueeze(-1))  # [B,T,C]，得到[B, T, d_model]
        # print("Shape of enc_out after embedding:", enc_out.shape) 

        B, T, d_model = enc_out.shape # 获取嵌入后的形状
        # print(f"B: {B}, T: {T}, d_model: {d_model}")
        '''
        if T != self.seq_len:
            if T > self.seq_len:
                enc_out = enc_out[:, :self.seq_len, :]
            else:
                # 如果 T 小于 self.seq_len，填充到 self.seq_len
                padding = torch.zeros(size=[B, self.seq_len - T, d_model]).to(enc_out.device)
                enc_out = torch.cat([enc_out, padding], dim=1)
        '''
        

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        enc_out = self.projection(torch.mean(enc_out,dim=1).unsqueeze(-1)).permute(0,2,1)
        # 确保 enc_out 形状为 [B, N, d_model]
        # enc_out = enc_out.permute(0, 2, 1)  # [B, T, d_model] -> [B, d_model, T]
        # enc_out = enc_out.reshape(B, -1, d_model)  # [B, d_model, T] -> [B, N, d_model]

        # 改Decoder
        L_pred = tp_to_predict.shape[-1]
        enc_out = enc_out.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        time_steps_to_predict = tp_to_predict.view(x_enc.shape[0], 1, L_pred, 1).repeat(1, enc_out.shape[1], 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        enc_out = torch.cat([enc_out, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(enc_out).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs
