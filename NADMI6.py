import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ATT import JCA
from encoder import build_backbone
class FEI(nn.Module):
    def __init__(self,inc):
        super(FEI,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inc,inc,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inc*3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self,x1,x2,x):
        f=self.conv1(x)
        f1=x1*f+x1
        f2=x2*f+x2
        F=torch.cat((f,f1,f2),dim=1)

        out=self.conv2(F)
        return out
class MOC(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_cfg=None):
        super(MOC, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )
        self.convfinal=nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        """Forward function."""
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        s1, s2 = torch.chunk(flow, 2, dim=1)
        x1_feat = torch.abs(self.warp(x1, s1) - x2)
        x2_feat = torch.abs(self.warp(x2, s2) - x1)
        output = torch.cat((x1_feat,x2_feat),dim=1)
        output = self.convfinal(output)
        return output
    def warp(self,x,flow):
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output




class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)




class MSI(nn.Module):
    def __init__(self,backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3,normal_init=True):
        super(MSI, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)


        self.cross1 = JCA(512)
        self.cross2 = JCA(256)
        self.cross3 = JCA(128)
        # self.cross4 = JCA(64)

        self.MOC1 = MOC(in_channels=512)
        self.MOC2 = MOC(in_channels=256)
        self.MOC3 = MOC(in_channels=128)
        self.MOC4 = MOC(in_channels=64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.FEI1 = FEI(512)
        self.FEI2 = FEI(256)
        self.FEI3 = FEI(128)
        self.FEI4 = FEI(64)

        self.deco1=nn.Sequential(
            nn.Conv2d(64*2,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(64*2,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(64*2,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.predict=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=2,kernel_size=1,stride=1)
        )






    def forward(self, imgs1, imgs2):
        c4, c1, c2, c3 = self.backbone(imgs1)
        c4_img2, c1_img2, c2_img2, c3_img2 = self.backbone(imgs2)
        cur1_1, cur2_1 = self.cross1(c4, c4_img2)  # 512
        cur1_2, cur2_2 = self.cross2(c3, c3_img2)  # 256
        cur1_3, cur2_3 = self.cross3(c2, c2_img2)  # 128

        cross_result1s = self.MOC1(cur1_1, cur2_1)
        cross_result2s = self.MOC2(cur1_2, cur2_2)
        cross_result3s = self.MOC3(cur1_3, cur2_3)
        cross_result4s = self.MOC4(c1, c1_img2)
        cur1_4, cur2_4 = c1, c1_img2


        feio1=self.FEI1(cross_result1s,cur1_1,cur2_1)
        feio2 = self.FEI2(cross_result2s, cur1_2, cur2_2)
        feio3 = self.FEI3(cross_result3s, cur1_3, cur2_3)
        feio4 = self.FEI4(cross_result4s, cur1_4, cur2_4)


        f512_256=self.deco1(torch.cat((feio1,feio2),dim=1))
        f512_256 = self.up2(f512_256)
        f256_128=self.deco2(torch.cat((feio3,f512_256),dim=1))
        f256_128=self.up2(f256_128)
        f128_64=self.deco3(torch.cat((feio4,f256_128),dim=1))
        f128_64=self.up2(f128_64)
        out_1=self.predict(f128_64)
        map = []
        map.append(out_1)
        return map

if __name__ == '__main__':
    net=MSI()
    a=torch.ones((1,3,256,256))
    b=torch.ones((1,3,256,256))
    c=net(a,b)
    print(c[0].shape)
