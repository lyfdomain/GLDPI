import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_size, input_size2, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),  # 输入层
            nn.BatchNorm1d(2048),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 512),  # 隐藏层1
            nn.BatchNorm1d(512),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_size2, 2048),  # 隐藏层1
            nn.BatchNorm1d(2048),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 512),  # 隐藏层2
            nn.BatchNorm1d(512),
            nn.Tanh(),
        )

    def forward(self, x, y):
        encoded = self.encoder(x)
        decoded = self.decoder(y)
        return encoded, decoded



class GBALoss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, e, sr, f, sp, dti, cof):  # 定义前向的函数运算即可
        # S = get_att_dis(e, e)
        Sd = torch.mm(e, e.T)
        sn = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)),
                      torch.sqrt_(torch.sum(e.mul(e), dim=1).view(1, torch.sum(e.mul(e), dim=1).shape[0])))
        # print(S.size(), sr.size())
        SN = torch.div(Sd, sn)
        los1 = torch.sum((SN - sr) ** 2) / (SN.shape[0] ** 2)
        Sp = torch.mm(f, f.T)
        snp = torch.mm(torch.sqrt_(torch.sum(f.mul(f), dim=1).view(torch.sum(f.mul(f), dim=1).shape[0], 1)),
                       torch.sqrt_(torch.sum(f.mul(f), dim=1).view(1, torch.sum(f.mul(f), dim=1).shape[0])))
        # print(S.size(), sr.size())
        SNp = torch.div(Sp, snp)
        los2 = torch.sum((SNp - sp) ** 2) / (SNp.shape[0] ** 2)
        S3 = torch.mm(e, f.T)
        sn3 = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)),
                       torch.sqrt_(torch.sum(f.mul(f), dim=1).view(1, torch.sum(f.mul(f), dim=1).shape[0])))
        # print(S.size(), sr.size())
        SN3 = torch.div(S3, sn3)
        los3 = torch.sum((SN3 - dti) ** 2 * cof) / (torch.sum(cof))
        los = 1 * los1 + 1 * los2 + 0.33 * los3

        return los
