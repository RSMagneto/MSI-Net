import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from encoder import  PixelAttention
import torch.nn.functional as F
class SPAjointAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPAjointAtt,self).__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 2, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 2, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        x=q1.unsqueeze(dim=2)
        y =q2.unsqueeze(dim=2)
        pattn = torch.cat([x, y], dim=2)  # B, C, 2, H, W
        pattn = Rearrange('b c t h w -> b (c t) h w')(pattn)
        q = pattn.view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2
        return  out1, out2

class SPECrossAtt(nn.Module):
    def __init__(self, in_channels):
        super(SPECrossAtt,self).__init__()
        self.in_channels = in_channels
        self.q_spe1=nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
        self.k_spe1=nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
        self.v_spe1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
        self.q_spe2 = nn.Conv2d(in_channels, in_channels , kernel_size=1, stride=1)
        self.k_spe2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v_spe2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bata =nn.Parameter(torch.zeros((1)))
        self.softmax=nn.Softmax(dim=-1)
        self.convspe1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )
        self.convspe2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )
    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q_spe1 = self.q_spe1(input1).view(batch_size, channels, -1)
        k_spe1 = self.k_spe1(input1).view(batch_size, channels, -1).permute(0, 2, 1)
        v_spe1 = self.v_spe1(input1).view(batch_size, channels, -1)
        q_spe2 = self.q_spe2(input2).view(batch_size, channels, -1)
        k_spe2 = self.k_spe2(input2).view(batch_size, channels, -1).permute(0, 2, 1)
        v_spe2 = self.v_spe2(input2).view(batch_size, channels, -1)
        energy1 = torch.bmm(q_spe2, k_spe1)
        energy_new1 = torch.max(energy1, -1, keepdim=True)[0].expand_as(energy1) - energy1
        attention1 = self.softmax(energy_new1)
        # print('attention大小：', attention.shape)
        out1 = torch.bmm(attention1, v_spe1)
        out1 = out1.view(batch_size, channels, height, width)
        # print('矩阵大小2：', out.shape)
        outspe1 = self.bata * out1 + input1

        energy2 = torch.bmm(q_spe1, k_spe2)
        energy_new2 = torch.max(energy2, -1, keepdim=True)[0].expand_as(energy2) - energy2
        attention2 = self.softmax(energy_new2)
        # print('attention大小：', attention.shape)
        out2 = torch.bmm(attention2, v_spe2)
        out2 = out2.view(batch_size, channels, height, width)
        # print('矩阵大小2：', out.shape)
        outspe2 = self.bata * out2 + input2
        return  outspe1, outspe2


class JCA(nn.Module):
    def __init__(self,in_channels):
        super(JCA,self).__init__()
        self.spe = SPECrossAtt(in_channels=in_channels)
        self.spa = SPAjointAtt(in_channels=in_channels, out_channels=in_channels)

    def forward(self,x1,x2):
        spe1, spe2 = self.spe(x1, x2)
        spa1, spa2 = self.spa(spe1, spe2)

        return spa1,spa2
