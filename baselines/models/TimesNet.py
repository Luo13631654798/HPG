import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from ..layers.Embed import DataEmbedding
from ..layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            print(f"Period: {period}")
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            print(f"Length: {length}, Period: {period}")

            # 确保长度与 period 兼容
            # if length % period != 0:
                # raise ValueError(f"Length {length} is not compatible with period {period}")
            # 计算总大小以确保 reshape 操作的一致性
            total_size_before = out.numel()
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            print(f"out shape after permute: {out.shape}")  # 添加调试信息

            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            total_size_after = out.numel()
            print(f"total_size_before: {total_size_before}, total_size_after: {total_size_after}")  # 添加调试信息
            if total_size_before != total_size_after:
                raise ValueError(f"Mismatch in size after reshape: before={total_size_before}, after={total_size_after}")

            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # add linear layer
        # self.linear_layer = nn.Linear(41, self.configs.enc_in)

        
        self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
        
        
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, configs.d_model - 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model), # 线性层
            nn.ReLU(inplace=True), # 激活函数relu
            nn.Linear(configs.d_model, configs.d_model), # 线性层
            nn.ReLU(inplace=True), # 激活函数relu
            nn.Linear(configs.d_model, 1) # 线性层
        )
    
    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        # Normalization from Non-stationary Transformer
        # self, tp_to_predict, observed_data, observed_tp, observed_mask
        # tp_to_predict: 要预测的时刻 [B, L_Pred]
        # observed_data: 观测值 [B, T, V]
        # observed_mask: 掩码 [B, T, V] 一般不用
        # observed_tp: 观测时刻 [B, T, 1]

        # forcast改名为forcasting
        # 参数名统一
        # 去Time Series Library找原始输入格式 跟这里的参数对应
        # Normalization from Non-stationary Transformer
        #  [B T V]
        means = observed_data.mean(1, keepdim=True).detach()
        x_enc = observed_data - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape
        print(f"x_enc11 shape: {x_enc.shape}, observed_tp shape: {observed_tp.shape}") # [32 96 12]  [32 96]
        ### add the content ###
        padding_len = self.seq_len - x_enc.shape[1]
        padding = torch.zeros(size=[x_enc.shape[0], padding_len, x_enc.shape[2]]).to(observed_data.device)
        x_enc = torch.cat([x_enc, padding], dim=1)
        print(f"x_enc22 shape: {x_enc.shape}, observed_tp shape: {observed_tp.shape}") # [32 98 12]  [32 96]
        padding_t = torch.zeros(size=[x_enc.shape[0], padding_len]).to(observed_data.device)
        observed_tp = torch.cat([observed_tp, padding_t], dim=1)

        # 确认传递给模型的输入形状
        # 32是batch 98是seq_len  12是ndim
        # x_enc 32 98 12   observed_tp 32 98
        # print(f"x_enc shape: {x_enc.shape}, observed_tp shape: {observed_tp.shape}") # [32 98 12]  [32 98]

        # x_enc = self.linear_layer(x_enc)

        # embedding 主要问题
        enc_out = self.enc_embedding(x_enc, observed_tp.unsqueeze(-1))  # [B,T,C]
        print(f"enc_out shape: {enc_out.shape}")
        # enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            # 0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        enc_out = enc_out[:, :-1]
        # batchsize 变量数 d_model

        # 改Decoder
        L_pred = tp_to_predict.shape[-1]
        enc_out = enc_out.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = tp_to_predict.view(x_enc.shape[0], 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        enc_out = torch.cat([enc_out, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(enc_out).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs
