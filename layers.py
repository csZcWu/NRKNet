import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        if not self.no_spatial:
            x_out = 1 / 3 * (self.hw(x) + self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1,
                                                                                              3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        else:
            x_out = 1 / 2 * (self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        return x_out


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          padding_mode='reflect'))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()


def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window


class GCM_Multi_Basics(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='FG', channels=3, num_groups=4):
        super(GCM_Multi_Basics, self).__init__()
        self.channels = channels
        init_kernel_size = 3
        kernel_size = 3
        increment = (max_kernel_size - kernel_size) / (num_kernels - 2)
        sigmas = np.linspace(0.2, 1, num_kernels)
        weight = torch.zeros(num_groups, num_kernels + 1, 1, max_kernel_size, max_kernel_size).cuda()
        for g in range(num_groups):
            for i in range(num_kernels):
                pad = int((max_kernel_size - kernel_size) / 2)
                weight[g, i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=g + sigmas[i]).cuda(),
                                          [pad, pad, pad, pad])).squeeze(0)
                kernel_size = np.min([max_kernel_size, int(init_kernel_size + (i * increment) // 2 * 2)])
            pad = int((max_kernel_size - 1) / 2)
            weight[g, 0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(), [pad, pad, pad, pad])).squeeze(0)
        # kernel = torch.FloatTensor(np.repeat(weight, self.channels, axis=0)).cuda()
        if mode == 'TG':
            self.weight = weight
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels + 1, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = weight
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)
        print(self.weight.shape)

    def __call__(self, x, weight):
        # temp = self.weight.detach().unsqueeze(1).cpu().numpy()
        # for i in range(len(temp)//3):
        #     cv2.imwrite('kernels1/TG/' + str(i) + '.png', temp[i, 0, 0] * 255. * 1)
        batch = x.shape[0]
        result = torch.FloatTensor().cuda()
        for b in range(batch):
            weighted_kernels = weight[b] * self.weight
            kernels = torch.sum(weighted_kernels, dim=0)
            y = F.conv2d(x[b].unsqueeze(0), torch.repeat_interleave(kernels, self.channels, dim=0),
                         padding=self.padding, groups=self.channels)
            result = torch.cat([result, y], dim=0)
            # print(y.shape)
            # x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        # print(result.shape)
        return result


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        # print(hidden.shape, c.shape)
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv(ch_in, ch_out, 3, 1, norm=True)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, hidden_state)
        y = self.upsample(h)
        return (y * x_res) + y, h, c


class SqueezeAttentionBlock_no_lstm(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock_no_lstm, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv(ch_in, ch_out, 3, 1, norm=True)
        self.conv_atten = BasicConv(ch_in, ch_out, 5, 1, norm=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h= self.conv_atten(y)
        y = self.upsample(h)
        return (y * x_res) + y


class SumLayer(nn.Module):
    def __init__(self, num_kernels=21, trainable=False):
        super(SumLayer, self).__init__()
        self.conv = nn.Conv2d((num_kernels) * 3, 3, 1, bias=True)

    def forward(self, x):
        return self.conv(x)


class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()

    def forward(self, x, y):
        return x * torch.repeat_interleave(y, 3, dim=1)
        # print(y.shape,x.shape)
        # return x * torch.cat([y, y, y], dim=1)


class GCM_V0(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='FG', channels=3):
        super(GCM_V0, self).__init__()
        self.channels = channels
        init_kernel_size = 3
        kernel_size = 3
        increment = (max_kernel_size - kernel_size) / (num_kernels - 2)
        sigmas = np.linspace(0.5, 1, num_kernels)
        kernel_sizes = [np.min([max_kernel_size, int(init_kernel_size + (i * increment) // 2 * 2)]) for i in
                        range(num_kernels)]
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, 3, size, 1, padding=int((size - 1) / 2), padding_mode='reflect', groups=channels, bias=False)
             for size in kernel_sizes])

        if not mode == 'TR':
            for i in range(len(self.convs)):
                a = nn.Parameter(gen_gaussian_kernel(kernel_sizes[i], sigma=sigmas[i]))
                a = nn.Parameter(torch.repeat_interleave(a, 3, dim=0), requires_grad=False)
                self.convs[i].weight = a
            if mode == 'FG':
                for conv in self.convs:
                    conv.weight.requires_grad = False
            else:
                for conv in self.convs:
                    conv.weight.requires_grad = True

    def forward(self, x):
        return torch.cat([x, torch.cat([conv(x) for conv in self.convs], dim=1)], dim=1)


e = 2.71828


class GCM_V1(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='FG', channels=3):
        super(GCM_V1, self).__init__()
        self.c = channels
        self.linespace_3 = nn.Parameter(torch.linspace(-10, 10, 21), requires_grad=False).cuda() ** 2
        # self.linespace_5 = nn.Parameter(torch.linspace(-2, 2, 5), requires_grad=False).cuda()

    def forward(self, x, sigmas):
        batch_size, in_planes, height, width = x.size()
        nk = len(sigmas[0])
        res = []
        for i in range(batch_size):
            sigmas_i = sigmas[i]
            blurry_img = x[i].unsqueeze(0)
            res1 = [blurry_img]
            for j in range(nk):
                # if j <= nk // 2:
                #     kernel_1d = torch.exp(-self.linespace_3 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                #     s = torch.sum(kernel_1d)
                #     kernel_1d = kernel_1d / s
                #     kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                #     kernel = torch.repeat_interleave(kernel, 3, dim=0)
                #     res1.append(F.conv2d(F.pad(res1[-1], [1, 1, 1, 1], mode='reflect'), kernel, groups=self.c))
                # else:
                #     kernel_1d = torch.exp(-self.linespace_5 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                #     s = torch.sum(kernel_1d)
                #     kernel_1d = kernel_1d / s
                #     kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                #     kernel = torch.repeat_interleave(kernel, 3, dim=0)
                #     res1.append(F.conv2d(F.pad(res1[-1], [2, 2, 2, 2], mode='reflect'), kernel, groups=self.c))
                kernel_1d = torch.exp(-self.linespace_3 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                # kernel_1d = torch.pow(e, -self.linespace_3 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                s = torch.sum(kernel_1d)
                kernel_1d = kernel_1d / s
                kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                kernel = torch.repeat_interleave(kernel, 3, dim=0)
                res1.append(F.conv2d(F.pad(res1[-1].detach(), [10, 10, 10, 10], mode='reflect'), kernel, groups=self.c))

            res.append(torch.cat(res1, dim=1))
        return torch.cat(res, dim=0)


class GCM_V2(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='FG', channels=3):
        super(GCM_V2, self).__init__()
        self.c = channels
        self.linespace_3 = nn.Parameter(torch.linspace(-10, 10, 21), requires_grad=False).cuda() ** 2
        # self.linespace_5 = nn.Parameter(torch.linspace(-2, 2, 5), requires_grad=False).cuda()

    def forward(self, x, sigmas):
        batch_size, in_planes, height, width = x.size()
        nk = len(sigmas[0])
        res = []
        for i in range(batch_size):
            sigmas_i = sigmas[i]
            blurry_img = x[i].unsqueeze(0)
            res1 = [blurry_img]
            for j in range(nk):
                # if j <= nk // 2:
                #     kernel_1d = torch.exp(-self.linespace_3 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                #     s = torch.sum(kernel_1d)
                #     kernel_1d = kernel_1d / s
                #     kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                #     kernel = torch.repeat_interleave(kernel, 3, dim=0)
                #     res1.append(F.conv2d(F.pad(res1[-1], [1, 1, 1, 1], mode='reflect'), kernel, groups=self.c))
                # else:
                #     kernel_1d = torch.exp(-self.linespace_5 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                #     s = torch.sum(kernel_1d)
                #     kernel_1d = kernel_1d / s
                #     kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                #     kernel = torch.repeat_interleave(kernel, 3, dim=0)
                #     res1.append(F.conv2d(F.pad(res1[-1], [2, 2, 2, 2], mode='reflect'), kernel, groups=self.c))
                kernel_1d = torch.exp(-self.linespace_3 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                # kernel_1d = torch.pow(e, -self.linespace_3 ** 2 / (2 * sigmas_i[j] ** 2)).unsqueeze(1)
                s = torch.sum(kernel_1d)
                kernel_1d = kernel_1d / s
                kernel = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
                kernel = torch.repeat_interleave(kernel, 3, dim=0)
                res1.append(F.conv2d(F.pad(res1[-1].detach(), [10, 10, 10, 10], mode='reflect'), kernel, groups=self.c))

            res.append(torch.cat(res1, dim=1))
        return torch.cat(res, dim=0)


class KernelPredictingBranch(nn.Module):
    def __init__(self, num_kernels=21):
        super(KernelPredictingBranch, self).__init__()


class SigmasPredictor(nn.Module):
    def __init__(self, in_planes, ratios, K, init_weight=True):
        super(SigmasPredictor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return 0.01 + F.sigmoid(x) * 10


if __name__ == '__main__':
    # kernel_sizes = sorted([i for i in range(3, 22, 2)]*2)
    # kernel_sizes.append(21)
    # a = nn.ModuleList(
    #     [nn.Conv2d(3, 3, kernel_sizes[i], 1, padding=int((kernel_sizes[i] - 1) / 2), padding_mode='reflect', groups=3).cuda() for i in
    #      range(len(kernel_sizes))])
    GCM_V0().cuda()
    # gcm_v1 = GCM_V0().cuda()
    # r = torch.randn((1, 3, 1280, 720)).cuda()
    # macs, para = thop.profile(gcm_v1, inputs=(r,))
    # print(macs / 1e9, para / 1e6)
