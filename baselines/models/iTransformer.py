import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, configs.d_model - 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(configs.d_model, 1)
        )
    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
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
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        padding_len = self.seq_len - x_enc.shape[1]
        padding = torch.zeros(size=[x_enc.shape[0], padding_len, x_enc.shape[2]]).to(observed_data.device)
        x_enc = torch.cat([x_enc, padding], dim=1)
        padding_t = torch.zeros(size=[x_enc.shape[0], padding_len]).to(observed_data.device)
        observed_tp = torch.cat([observed_tp, padding_t], dim=1)

        # Embedding
        enc_out = self.enc_embedding(x_enc, observed_tp.unsqueeze(-1))
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :-1] # 保证Shape: [B， V， d_model维]

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


