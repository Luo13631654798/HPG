import torch
import torch.nn as nn
from ..layers.AMS import AMS
from ..layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size_list=self.patch_size_list, noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection))
        
        '''
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )
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

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        # self, tp_to_predict, observed_data, observed_tp, observed_mask
        # self, x
        # add the content
        '''
        means = observed_data.mean(1, keepdim=True).detach()
        x_enc = observed_data - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        '''
        x_enc = observed_data
        

        _, _, N = x_enc.shape
        ### add the content ###
        padding_len = self.seq_len - x_enc.shape[1]
        padding = torch.zeros(size=[x_enc.shape[0], padding_len, x_enc.shape[2]]).to(observed_data.device)
        x_enc = torch.cat([x_enc, padding], dim=1)
        padding_t = torch.zeros(size=[x_enc.shape[0], padding_len]).to(observed_data.device)
        observed_tp = torch.cat([observed_tp, padding_t], dim=1)


        balance_loss = 0
        '''
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        '''
        # norm
        
        out = self.start_fc(x_enc.unsqueeze(-1))


        batch_size = x_enc.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1,3)
        out = torch.mean(out, dim=-2) 
        # Batchsize, Var = 12, D_model
        '''
        out = self.projections(out).transpose(2, 1)
        '''



        # 改Decoder
        L_pred = tp_to_predict.shape[-1]
        enc_out = out.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, F)
        # print(h.shape, time_steps_to_predict.shape)
        time_steps_to_predict = tp_to_predict.view(x_enc.shape[0], 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)  # (B, N, Lp, F_te)

        enc_out = torch.cat([enc_out, te_pred], dim=-1)  # (B, N, Lp, F)

        # (B, N, Lp, F) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        outputs = self.decoder(enc_out).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        # # denorm
        # if self.revin:
        #     out = self.revin_layer(out, 'denorm')

        return outputs

        
        


