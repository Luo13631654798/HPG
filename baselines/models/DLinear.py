import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.c_out = configs.enc_in

        '''
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        '''
        self.te_scale = nn.Linear(1, 1) # 线性层，用于时间嵌入的缩放
        self.te_periodic = nn.Linear(1, configs.d_model - 1) # 线性层，用于时间嵌入的周期性部分


        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model), # 线性层
            nn.ReLU(inplace=True), # relu激活函数
            nn.Linear(configs.d_model, configs.d_model), # 线性层
            nn.ReLU(inplace=True), # relu激活函数
            nn.Linear(configs.d_model, 1) # 线性层，输出为1
        )

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)，输入时间序列
        out1 = self.te_scale(tt) # 时间嵌入的缩放
        out2 = torch.sin(self.te_periodic(tt)) # 时间嵌入的周期性部分
        return torch.cat([out1, out2], -1) # 将缩放和周期性部分拼接

    def forecasting(self, x):
        # Encoder
        # return self.encoder(x_enc)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        
        # Prepare input for new decoder
        L_pred = x.shape[-1]
        x = x.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        time_steps_to_predict = torch.arange(L_pred).view(1, 1, L_pred, 1).repeat(x.shape[0], x.shape[1], 1, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        x = torch.cat([x, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(x).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs
        # return x.permute(0, 2, 1)

