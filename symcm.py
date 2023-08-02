import torch
import torch.nn as nn
import torch.nn.functional as F


class FullRKR(nn.Module):
    def __init__(self, num_blurred_version=21, kernel_size=3):
        super(FullRKR, self).__init__()
        self.num_blurred_version = num_blurred_version
        self.ks = kernel_size
        self.kernel_size = kernel_size
        self.kernel_padding = (self.kernel_size - 1)//2
        self.kernels = nn.Parameter(torch.rand(num_blurred_version * 3, 1, 1, self.kernel_size), requires_grad=True)

    def forward(self, x, dilation=1):
        k_xs = F.softmax((self.kernels+self.kernels.flip(-1))/2, dim=3)
        k_ys = k_xs.view((self.num_blurred_version * 3, 1, self.ks, 1))
        # x = F.pad(x, [self.kernel_padding] * 4, mode='reflect')
        # print(self.kernels.shape, k_xs.shape, k_ys.shape, x.shape)
        res = [x]
        for i in range(0, self.num_blurred_version * 3, 3):
            c_x = F.conv2d(F.pad(res[-1], [self.kernel_padding + dilation - 1] * 4, mode='reflect'), k_xs[i:i + 3],
                           groups=3, dilation=(dilation, dilation))
            res.append(F.conv2d(c_x, k_ys[i:i + 3], groups=3, dilation=(dilation, dilation)))
        return torch.cat(res, dim=1)


class NonRecursive(nn.Module):
    def __init__(self, num_blurred_version=21):
        super(NonRecursive, self).__init__()
        self.num_blurred_version = num_blurred_version
        kernel_sizes = [i for i in range(3, 3 + num_blurred_version * 2, 2)]
        self.kernels = [
            nn.Parameter(
                F.softmax(torch.rand(3, 1, 1, kernel_size), dim=3).cuda()[:, :, :, :int((kernel_size + 1) / 2)],
                requires_grad=True) for kernel_size in kernel_sizes]
        self.kernel_paddings = [int((kernel_size + 1) / 2) - 1 for kernel_size in kernel_sizes]

    def forward(self, x):
        res = [x]
        for (kernel, kernel_padding) in zip(self.kernels, self.kernel_paddings):
            k_x = F.softmax(F.pad(kernel, pad=[0, kernel_padding, 0, 0], mode='reflect'), dim=3)

            # print(k_x.shape)
            k_y = k_x.view((3, 1, 2 * kernel_padding + 1, 1))
            # print(k_y.shape)
            c_x = F.conv2d(F.pad(res[-1], [kernel_padding] * 4, mode='reflect'), k_x, groups=3)

            res.append(F.conv2d(c_x, k_y, groups=3))
            # print(res[-1].shape)
        return torch.cat(res, dim=1)


class NonSeparable(nn.Module):
    def __init__(self, num_blurred_version=21):
        super(NonSeparable, self).__init__()
        self.num_blurred_version = num_blurred_version
        kernel_sizes = [i for i in range(3, 3 + num_blurred_version * 2, 2)]
        # self.kernels = [
        #     nn.Parameter(
        #         F.softmax(torch.rand(3, 1, 1, kernel_size), dim=3).cuda()[:, :, :, :int((kernel_size + 1) / 2)],
        #         requires_grad=True) for kernel_size in kernel_sizes]
        self.kernels = [
            nn.Parameter(torch.rand((3, 1, (kernel_size + 1) // 2, (kernel_size + 1) // 2)).cuda(), requires_grad=True)
            for kernel_size in kernel_sizes]
        self.kernel_paddings = [int((kernel_size + 1) / 2) - 1 for kernel_size in kernel_sizes]

    def forward(self, x):
        res = [x]
        for (kernel, kernel_padding) in zip(self.kernels, self.kernel_paddings):
            kernel = kernel + torch.flip(torch.rot90(kernel, 1, dims=[2, 3]), dims=[2])
            kernel = F.pad(kernel, pad=[0, kernel_padding, 0, kernel_padding], mode='reflect').view(
                (3, 1, (2 * kernel_padding + 1) * (2 * kernel_padding + 1)))
            k = F.softmax(kernel, dim=2).view((3, 1, (2 * kernel_padding + 1), (2 * kernel_padding + 1)))
            c_x = F.conv2d(F.pad(x, [kernel_padding] * 4, mode='reflect'), k, groups=3)
            res.append(c_x)
        return torch.cat(res, dim=1)


if __name__ == '__main__':
    skcm = FullRKR(21)
    a = torch.rand((1, 3, 256, 256))
    b = skcm(a, 3)
    print(b.shape)
    # # c = nn.Conv2d(3, 9, (5, 5), groups=3)
    # # print(c.weight.shape)
    # # a = [i for i in range(3, 45, 2)]
    # # print(a)
    # # print(len(a))
    # a = torch.rand((1, 3, 256, 256))
    # b = nn.Conv2d(3, 66, 3, stride=(1, 1), padding=(3, 3), dilation=(3, 3), groups=3, bias=False)
    # c = b(a)
    # print(c.shape)
