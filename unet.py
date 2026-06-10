import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder
        self.ec1=nn.Conv2d(6, 16, 3, padding=1)
        self.ebn1=nn.BatchNorm2d(16)
        self.erl1=nn.ReLU(inplace=True)
        self.emp1=nn.MaxPool2d(2)

        self.ec2=nn.Conv2d(16, 32, 3, padding=1)
        self.ebn2=nn.BatchNorm2d(32)
        self.erl2=nn.ReLU(inplace=True)
        self.emp2=nn.MaxPool2d(2)

        self.ec3=nn.Conv2d(32, 32, 3, padding=1) 
        self.ebn3=nn.BatchNorm2d(32)
        self.erl3=nn.ReLU(inplace=True)
        self.emp3=nn.MaxPool2d(2)

        self.ec4=nn.Conv2d(32, 32, 3, padding=1) 
        self.ebn4=nn.BatchNorm2d(32)
        self.erl4=nn.ReLU(inplace=True)
        self.emp4=nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.bottleneck_conv_bn=nn.BatchNorm2d(32)
        self.bottleneck_relu = nn.ReLU(inplace=True)   

        #decoder
        self.dct1=nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dc1=nn.Conv2d(48, 48, 3, padding=1) 
        self.dbn1=nn.BatchNorm2d(48)
        self.drl1=nn.ReLU(inplace=True)

        self.dct2=nn.ConvTranspose2d(48, 16, 2, stride=2)
        self.dc2=nn.Conv2d(48, 48, 3, padding=1) 
        self.dbn2=nn.BatchNorm2d(48)
        self.drl2=nn.ReLU(inplace=True)

        self.dct3=nn.ConvTranspose2d(48, 16, 2, stride=2)
        self.dc3=nn.Conv2d(48, 48, 3, padding=1) 
        self.bn3=nn.BatchNorm2d(48)

        self.dct4=nn.ConvTranspose2d(48, 3, 2, stride=2)


    def forward(self, input):
        #encoder
        #print(input.shape)
        fec1 = self.ec1(input)
        fec_bn1 = self.ebn1(fec1)
        fec_bn_rl1 = self.erl1(fec_bn1)
        fec_bn_rl_mp1=self.emp1(fec_bn_rl1)
        #print(fec_bn_rl_mp1.shape)

        fec2=self.ec2(fec_bn_rl_mp1)
        fec_bn2=self.ebn2(fec2)
        fec_bn_rl2 = self.erl2(fec_bn2)
        fec_bn_rl_mp2=self.emp2(fec_bn_rl2)
        #print(fec_bn_rl_do_mp2.shape)

        fec3=self.ec3(fec_bn_rl_mp2)
        fec_bn3=self.ebn3(fec3)
        fec_bn_rl3=self.erl3(fec_bn3)
        fec_bn_rl_mp3=self.emp3(fec_bn_rl3)
        #print(fec_bn_rl_mp3.shape)

        fec4=self.ec4(fec_bn_rl_mp3)
        fec_bn4=self.ebn4(fec4)
        fec_bn_rl4=self.erl4(fec_bn4)
        fec_bn_rl_mp4=self.emp4(fec_bn_rl4)
        #print(fec_bn_rl_mp4.shape)

        # bottleneck layer
        fbneck_c = self.bottleneck_conv(fec_bn_rl_mp4)
        fbneck_c_bn = self.bottleneck_conv_bn(fbneck_c)
        fbneck_c_bn_relu = self.bottleneck_relu(fbneck_c_bn)  
        #print(fbneck_c_bn_relu.shape)

        #decoder
        fdct1=self.dct1(fbneck_c_bn_relu)
        fdct1_concat=torch.cat([fdct1, fec_bn_rl4], dim=1)
        fdct1_concat_c1=self.dc1(fdct1_concat)
        fdct1_concat_c1_bn1=self.dbn1(fdct1_concat_c1)
        fdct1_concat_c1_bn1_rl1=self.drl1(fdct1_concat_c1_bn1)
        #print(fdct1_concat_bn1_rl1.shape)
           
        fdct2=self.dct2(fdct1_concat_c1_bn1_rl1)
        #print(fdct2.shape)
        fdct2_concat=torch.cat([fdct2, fec_bn_rl3], dim=1)
        fdct2_concat_c2=self.dc2(fdct2_concat)
        fdct2_concat_c2_bn2=self.dbn2(fdct2_concat_c2)
        fdct2_concat_c2_bn2_rl2=self.drl2(fdct2_concat_c2_bn2)
        #print(fdct2_concat_bn2_rl2.shape)

        fdct3=self.dct3(fdct2_concat_c2_bn2_rl2)
        fdct3_concat=torch.cat([fdct3, fec_bn_rl2], dim=1)
        fdct3_concat_c3 = self.dc3(fdct3_concat)
        fdct3_concat_c3_bn3=self.bn3(fdct3_concat_c3)
       
        #dfct_bn_rl_c_bn3 = self.bn31(fdct_bn_rl_c3)

        out=self.dct4(fdct3_concat_c3_bn3)

        return out