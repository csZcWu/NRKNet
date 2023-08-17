import time

from layers import *
from symcm import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NRKNet(nn.Module):
    """
    The version of fixed-point iteration with different iteration numbers in different scale
    Re-blurred source: the de-blurred image of last scale
    Number of scales: 3
    Number of iteration in each scale: [1,2,3]
    Number of times to estimate the blur kernel in each scale: 1
    Normalization form: None
    """

    def __init__(self, config):
        super(NRKNet, self).__init__()

        base_channel = 64
        num_res = config.net['num_res']
        num_kernels = config.net['num_kernels']
        in_ch = config.net['in_ch']
        self.SummationLayer = SumLayer(num_kernels + 1)
        self.MultiplyLayer = MultiplyLayer()
        self.GCM = FullRKR(num_kernels)
        self.APU = SqueezeAttentionBlock(base_channel, num_kernels + 1)

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.bottleneck = nn.Sequential(
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=3, relu=True, stride=1),
            EBlock(base_channel * 4, num_res)
        )

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 1, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel * 1, kernel_size=4, relu=True, stride=2, transpose=True)
        ])

    def forward(self, x, y=None, phase='train', scales=3):
        blurrys = []
        deblurred = []
        items=[]
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        xs = [x_4, x_2, x]
        if phase == 'train':

            y_2 = F.interpolate(y, scale_factor=0.5)
            y_4 = F.interpolate(y_2, scale_factor=0.5)
            ys = [y_4, y_2, y]
            h, c = self.APU.conv_atten.init_hidden(xs[0].shape[0], (xs[0].shape[-2] // 2, xs[0].shape[-1] // 2))
            dbd = xs[-1]
            for i in range(len(xs)):
                '''Feature Extract 0'''
                x_ = self.feat_extract[0](xs[i])
                res1 = self.Encoder[0](x_)

                '''Down Sample 1'''
                z = self.feat_extract[1](res1)
                res2 = self.Encoder[1](z)

                '''Down Sample 2'''
                z = self.feat_extract[2](res2)
                res3 = self.Encoder[2](z)

                '''Bottle Neck'''
                z = self.bottleneck(res3)

                '''Up Sample 2'''
                z = self.Decoder[0](res3 + z)
                z = self.feat_extract[3](z)

                '''Up Sample 1'''
                z = self.Decoder[1](z + res2)
                z = self.feat_extract[4](z)

                z = self.Decoder[2](z + res1)
                z_, h, c = self.APU(z, (h, c))
                db_result = xs[i]
                temp = xs[i]
                for j in range(scales - i):
                    temp = temp - self.SummationLayer(self.MultiplyLayer(self.GCM(temp), z_))
                    db_result = db_result + temp
                items.append(temp)
                dbd = dbd + F.interpolate(temp, scale_factor=2 ** (scales - i - 1), mode='bilinear')
                deblurred.append(db_result)
                blur = self.SummationLayer(self.MultiplyLayer(self.GCM(ys[i]), z_))
                blurrys.append(blur)
                h = F.interpolate(h, scale_factor=2, mode='bilinear')
                c = F.interpolate(c, scale_factor=2, mode='bilinear')
            deblurred.append(dbd)
            return deblurred, blurrys
        else:
            h, c = self.APU.conv_atten.init_hidden(xs[0].shape[0], (xs[0].shape[-2] // 2, xs[0].shape[-1] // 2))
            dbd = xs[-1]
            for i in range(len(xs)):
                '''Feature Extract 0'''
                x_ = self.feat_extract[0](xs[i])
                res1 = self.Encoder[0](x_)

                '''Down Sample 1'''
                z = self.feat_extract[1](res1)
                res2 = self.Encoder[1](z)

                '''Down Sample 2'''
                z = self.feat_extract[2](res2)
                res3 = self.Encoder[2](z)

                '''Bottle Neck'''
                z = self.bottleneck(res3)

                '''Up Sample 2'''
                z = self.Decoder[0](res3 + z)
                z = self.feat_extract[3](z)

                '''Up Sample 1'''
                z = self.Decoder[1](z + res2)
                z = self.feat_extract[4](z)

                z = self.Decoder[2](z + res1)
                z_, h, c = self.APU(z, (h, c))
                temp = xs[i]
                for j in range(scales - i):
                    temp = temp - self.SummationLayer(self.MultiplyLayer(self.GCM(temp), z_))
                items.append(temp)
                dbd = dbd + F.interpolate(temp, scale_factor=2 ** (scales - i - 1), mode='bilinear')
                if i < scales - 1:
                    h = F.interpolate(h, scale_factor=2, mode='bilinear')
                    c = F.interpolate(c, scale_factor=2, mode='bilinear')
            deblurred.append(
                xs[-1] + items[-1] + F.interpolate(items[-2], scale_factor=2, mode='bilinear') + F.interpolate(
                    items[-3],
                    scale_factor=4,
                    mode='bilinear'))
            return deblurred, blurrys


if __name__ == '__main__':
    import config as config
    from thop import profile
    kmlnet = NRKNet(config).cuda()
    a = torch.rand((1, 3, 1280, 720)).cuda()
    flops, params = profile(kmlnet, inputs=(a,None,'test'))
    print(flops/2, params)
