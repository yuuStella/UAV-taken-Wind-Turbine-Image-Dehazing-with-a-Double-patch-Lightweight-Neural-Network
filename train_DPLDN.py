import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
import lightdehazeNet2
import cv2
import numpy as np
import torch.utils.data
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models import vgg16
import torch.nn.functional as F
from math import exp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def train(args):

    ld_net1 = lightdehazeNet2.LightDehaze_Net().cuda()
    ld_net1.apply(weights_init)

    ld_net2 = lightdehazeNet2.LightDehaze_Net().cuda()
    ld_net2.apply(weights_init)

    training_data = image_data_loader.hazy_data_loader(args["train_original"],
                                                       args["train_hazy"])
    validation_data = image_data_loader.hazy_data_loader(args["train_original"],
                                                         args["train_hazy"], mode="val")
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=4,
                                                       pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=4,
                                                         pin_memory=True)

    criterion = nn.SmoothL1Loss(beta=1).cuda()

    # 优化器，学习率
    optimizer1 = torch.optim.Adam(ld_net1.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.2)

    optimizer2 = torch.optim.Adam(ld_net2.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.2)


    lr_list = []

    ld_net1.train()
    ld_net2.train()

    num_of_epochs = int(args["epochs"])
    # 构建 SummaryWriter
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    for epoch in range(num_of_epochs):
        print("-----")
        print("epoch:", epoch)
        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            H = hazy_image.size(2)
            # W = hazy_image.size(3)

            images_lv1 = Variable(hazy_image - 0.5).cuda()

            # 第二尺度图像输入
            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

            torchvision.utils.save_image(images_lv2_1, "./hazefree_edge/" + str(103) + ".jpg")

            # 将第二尺度的图片与第三尺度融合在一起后合并
            feature_lv2_1 = ld_net2(images_lv2_1)
            feature_lv2_2 = ld_net2(images_lv2_2)
            torchvision.utils.save_image(feature_lv2_1, "./hazefree_edge/" + str(109) + ".jpg")
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + images_lv1

            torchvision.utils.save_image(feature_lv2, "./hazefree_edge/" + str(107) + ".jpg")
            dehaze_image = ld_net1(feature_lv2)
            torchvision.utils.save_image(dehaze_image, "./hazefree_edge/" + str(108) + ".jpg")

            # dehaze_image = ld_net(hazy_image)

            # 原图求边缘
            torchvision.utils.save_image(hazefree_image, "./hazefree_edge/" + str(100) + ".jpg")

            image = cv2.imread("./hazefree_edge/" + str(100) + ".jpg", 0)

            image_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            absX = (cv2.convertScaleAbs(image_x))

            image_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            absY = cv2.convertScaleAbs(image_y)

            dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            # 去雾图求边缘
            torchvision.utils.save_image(dehaze_image, "./hazefree_edge/" + str(101) + ".jpg")

            image2 = cv2.imread("./hazefree_edge/" + str(101) + ".jpg", 0)

            image_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
            absX2 = (cv2.convertScaleAbs(image_x2))

            image_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
            absY2 = cv2.convertScaleAbs(image_y2)

            dst2 = cv2.addWeighted(absX2, 0.5, absY2, 0.5, 0)
            # 计算边缘损失并加入损失函数中
            loss2 = np.square(dst - dst2)
            loss2 = loss2.sum() / 20000000000


            # 加入SSIM损失
            ssim_loss = 1 - SSIM()(dehaze_image, hazefree_image)

            loss = 0.1 * loss2 + 0.84 * ssim_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(ld_net1.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm(ld_net2.parameters(), 0.1)
            optimizer1.step()
            optimizer2.step()

            if ((iteration + 1) % 10) == 0:
                print("Sum Loss at iteration", iteration + 1, ":", loss.item())
                print("Edge loss at iteration", iteration + 1, ":", loss2.item())
                print("Ssim loss at iteration", iteration + 1, ":", ssim_loss.item())
            if ((iteration + 1) % 200) == 0:
                torch.save(ld_net1.state_dict(), "trained_weights/" + "Epoch" + str(epoch) + "a8" + '.pth')
                torch.save(ld_net2.state_dict(), "trained_weights/" + "Epoch" + str(epoch) + "b8" + '.pth')

        # 学习率改变1
        current_lr1 = optimizer1.state_dict()['param_groups'][0]['lr']  # 当前学习率
        lr_list.append(current_lr1)
        print("current_lr1:", current_lr1)

        scheduler1.step()  # 调整学习率

        adjusted_lr1 = scheduler1.get_last_lr()
        print("adjusted_lr1:", adjusted_lr1)
        print("-----")

        # 2
        current_lr2 = optimizer2.state_dict()['param_groups'][0]['lr']  # 当前学习率
        lr_list.append(current_lr2)
        print("current_lr2:", current_lr2)

        scheduler2.step()  # 调整学习率

        adjusted_lr2 = scheduler2.get_last_lr()
        print("adjusted_lr2:", adjusted_lr2)
        print("-----")

        # Validation Stage
        for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            # dehaze_image = ld_net(hazy_image)
            H = hazy_image.size(2)
            W = hazy_image.size(3)

            images_lv1 = Variable(hazy_image - 0.5).cuda()

            # 第二尺度图像输入
            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

            torchvision.utils.save_image(images_lv2_1, "./hazefree_edge/" + str(103) + ".jpg")

            # 将第二尺度的图片与第三尺度融合在一起后合并
            feature_lv2_1 = ld_net2(images_lv2_1)
            feature_lv2_2 = ld_net2(images_lv2_2)
            torchvision.utils.save_image(feature_lv2_1, "./hazefree_edge/" + str(109) + ".jpg")
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + images_lv1

            torchvision.utils.save_image(feature_lv2, "./hazefree_edge/" + str(107) + ".jpg")
            dehaze_image = ld_net1(feature_lv2)
            torchvision.utils.save_image(dehaze_image, "./hazefree_edge/" + str(108) + ".jpg")

            # dehaze_image = ld_net(hazy_image)

            torchvision.utils.save_image(torch.cat((hazy_image, dehaze_image, hazefree_image), 0),
                                         "training_data_captures2/" + str(iter_val + 1) + ".jpg")

        torch.save(ld_net1.state_dict(), "trained_weights/" + "trained_LDNet18.pth")
        torch.save(ld_net2.state_dict(), "trained_weights/" + "trained_LDNet28.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-th", "--train_hazy", required=True, help="path to hazy training images")
    ap.add_argument("-to", "--train_original", required=True, help="path to original training images")
    ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
    ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")

    args = vars(ap.parse_args())

    train(args)








