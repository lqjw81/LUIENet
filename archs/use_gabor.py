import torch
import torch.nn as nn
from torchvision import transforms
# from testpython.gabor import getGabor, build_filters
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(

            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class SFDIM(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.Conv = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.WT = WT(n_feats)

    # def forward(self, x):
    def forward(self, x, y):
        mix = x + y
        # mix = x
        reconstructed_tensor = self.WT(mix)
        out = self.Conv(reconstructed_tensor)
        return out

def resize_image(img, size):
    transform = transforms.Resize(size)
    return transform(img)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    # if test
    x2 = resize_image(x2, (x1.shape[2], x1.shape[3]))
    x3 = resize_image(x3, (x1.shape[2], x1.shape[3]))
    x4 = resize_image(x4, (x1.shape[2], x1.shape[3]))

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class WT(nn.Module):
    def __init__(self, n_feats):
        super(WT, self).__init__()
        self.dw = DWT()
        self.iwt = IWT()
        self.wavelet = 'haar'  # 小波基函数（这里使用Haar小波）
        self.Conv_ll = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
        self.Conv_lh = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
        self.Conv_hl = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
        self.Conv_hh = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        # 对整个批次的张量进行小波变换
        ll, lh, hl, hh = self.dw(x)
        ll = self.Conv_ll(ll)
        lh = self.Conv_lh(lh)
        hl = self.Conv_hl(hl)
        hh = self.Conv_hh(hh)
        coffes = torch.cat((ll, lh, hl, hh), 1)
        reconstructed_tensor = self.iwt(coffes)
        return reconstructed_tensor


# Multi-branch Color Enhancement Modul
class MCEM(nn.Module):
    def __init__(self, in_channels, channels):
        super(MCEM, self).__init__()
        self.conv_first_r = nn.Conv2d(in_channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_g = nn.Conv2d(in_channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(in_channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.inception_r = InceptionDWConv2d(channels)
        self.inception_g = InceptionDWConv2d(channels)
        self.inception_b = InceptionDWConv2d(channels)

        self.conv_out_r = nn.Conv2d(channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_g = nn.Conv2d(channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d(channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x_1 = self.conv_first_r(x1)
        x_2 = self.conv_first_g(x2)
        x_3 = self.conv_first_b(x3)

        out_instance_r = self.inception_r(x_1)
        out_instance_g = self.inception_g(x_2)
        out_instance_b = self.inception_b(x_3)

        out_instance_r = self.conv_out_r(out_instance_r)
        out_instance_g = self.conv_out_g(out_instance_g)
        out_instance_b = self.conv_out_b(out_instance_b)

        mix = out_instance_r + out_instance_g + out_instance_b + x4

        out_instance = torch.cat((out_instance_r, out_instance_g, out_instance_b, mix), dim=1)
        return out_instance


class CC_UnderwaterColorCorrectionNet(nn.Module):

    def __init__(self):
        super(CC_UnderwaterColorCorrectionNet, self).__init__()

        # Convolution branch
        self.fusion = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fusion1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fusion2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fusion3 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        r_channel = input[:, 0:1, :, :]
        g_channel = input[:, 1:2, :, :]
        b_channel = input[:, 2:3, :, :]

        # Compute average values for each channel
        avg_r = torch.mean(r_channel)
        avg_g = torch.mean(g_channel)
        avg_b = torch.mean(b_channel)

        adj_m = max(avg_r, avg_g, avg_b)

        # Compute correction factors
        scale_r = (adj_m / avg_r)
        scale_g = (adj_m / avg_g)
        scale_b = (adj_m / avg_b)

        # Apply correction factors to each channel
        r_channel = r_channel * scale_r
        g_channel = g_channel * scale_g
        b_channel = b_channel * scale_b

        # Fusion of convolution results
        rgb = torch.cat((r_channel, g_channel, b_channel), dim=1)

        corrected_rgb = torch.zeros_like(rgb)

        for channel in range(3):
            channel_min = torch.min(rgb[:, channel, :, :])
            channel_max = torch.max(rgb[:, channel, :, :])
            corrected_channel = (rgb[:, channel, :, :] - channel_min) / (channel_max - channel_min)
            corrected_rgb[:, channel, :, :] = corrected_channel

        output = self.fusion(corrected_rgb)
        # output = self.fusion1(corrected_rgb)
        # output = self.fusion2(output)
        # output = self.fusion3(output)

        return output


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=1, band_kernel_size=7, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        # self.instance_1 = nn.InstanceNorm2d(10, affine=True)
        # self.instance_1 = nn.InstanceNorm2d(10, affine=True)
        self.instance = nn.InstanceNorm2d(in_channels, affine=True)
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return self.instance(torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1, ))



def build_filters(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    filters = []
    # ksize = [5, 7, 9, 11, 13, 15, 17]  # Gabor尺度，6个
    ksize = [7]  # Gabor尺度，6个
    # lamda = torch.tensor([np.pi / 2.0])  # 波长
    lamda = 1.5707963267948966
    pi = 3.141592653589793
    for theta in torch.arange(0, pi, pi / 4, device=device):  # Gabor方向，0°，45°，90°，135°，共四个
        for K in range(len(ksize)):
            sigma = ksize[K] // 3
            xmax, ymax = ksize[K] // 2, ksize[K] // 2
            xmin, ymin = -xmax, -ymax
            x, y = torch.meshgrid(torch.arange(xmin, xmax + 1, device=device),
                                  torch.arange(ymin, ymax + 1, device=device))
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
            gabor = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2)) * torch.cos(
                2 * pi * x_theta / lamda)
            gabor /= torch.sum(torch.abs(gabor))
            gabor = gabor.unsqueeze(0).unsqueeze(0)
            filters.append(gabor)
    return filters


def getGabor(x, filters):
    img = torch.mean(x, dim=1, keepdim=True)
    # 使用列表推导式将滤波器应用于输入图像
    accum_list = [F.relu(F.conv2d(img, kern.float().to(x.device), padding=kern.shape[2] // 2)) for kern in filters]
    # 在第二个维度上连接所有的张量
    accum_tensors = torch.cat(accum_list, dim=1)
    # 在第二个维度上连接输入图像和滤波后的张量
    out = torch.cat([x, accum_tensors], dim=1)

    return out


# MAIN-Net
class FIVE_APLUSNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=16):
        super(FIVE_APLUSNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)

        self.cc = CC_UnderwaterColorCorrectionNet()
        self.inception1 = InceptionDWConv2d(base_nf)
        self.inception2 = InceptionDWConv2d(base_nf)
        self.inception3 = InceptionDWConv2d(base_nf)

        # self.gabor = GaborConv2d(in_nc, base_nf, (3, 3))

        # self.conv2 = nn.Conv2d(in_nc * 9, base_nf, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_nc+4, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.fusion_mixer = SFDIM(base_nf)

        self.conv4 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.color_cer = MCEM(base_nf, base_nf)

        self.stage2 = PALayer(base_nf)

        self.act = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.filters = build_filters()

    def forward(self, x):
        cc = self.cc(x)
        out = self.conv1(cc)
        out_1 = self.inception1(out)
        out_1 = self.inception2(out_1)

        gabor_feature = getGabor(x, self.filters)
        gabor_feature = self.conv2(gabor_feature)
        out_2 = self.conv3(gabor_feature)

        mix_out = self.fusion_mixer(out_1, out_2)
        # mix_out = self.fusion_mixer(out_1)

        out_stage2 = self.act(mix_out)

        out_stage2_head = self.conv4(out_stage2)

        out_stage2 = self.conv5(out_stage2)
        out_stage2 = self.color_cer(out_stage2)
        out = self.stage2(out_stage2)
        out = self.act(out)
        out = self.conv6(out)

        return out, out_stage2_head
