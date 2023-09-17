import argparse
import glob
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from torch.autograd import Variable
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

def SRCNN_image(image_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file1', type=str, required=True, default="BLAH_BLAH/outputs/x2/best1.pth")
    parser.add_argument('--weights-file2', type=str, required=True, default="BLAH_BLAH/outputs/x2/best2.pth")
    parser.add_argument('--weights-file3', type=str, required=True, default="BLAH_BLAH/outputs/x2/best3.pth")
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model1 = SRCNN().to(device)
    model2 = SRCNN().to(device)
    model3 = SRCNN().to(device)

    state_dict1 = model1.state_dict()
    for n, p in torch.load(args.weights_file1, map_location=lambda storage, loc: storage).items():
        if n in state_dict1.keys():
            state_dict1[n].copy_(p)
        else:
            raise KeyError(n)

    state_dict2 = model2.state_dict()
    for n, p in torch.load(args.weights_file2, map_location=lambda storage, loc: storage).items():
        if n in state_dict2.keys():
            state_dict2[n].copy_(p)
        else:
            raise KeyError(n)

    state_dict3 = model3.state_dict()
    for n, p in torch.load(args.weights_file3, map_location=lambda storage, loc: storage).items():
        if n in state_dict3.keys():
            state_dict3[n].copy_(p)
        else:
            raise KeyError(n)

    model1.eval()
    model2.eval()
    model3.eval()

    image = pil_image.open(image_file).convert('RGB')

    # image_width = (image.width // args.scale) * args.scale
    # image_height = (image.height // args.scale) * args.scale
    # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    # image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    # image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    # image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        H = y.size(2)
        W = y.size(3)
        images_lv1 = Variable(y - 0.5).cuda()
        # 第二尺度图像输入
        images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
        images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

        # 第三图尺度图像输入
        images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
        images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
        images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
        images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]
        # 由细到粗，先输入最细尺度，输入四个小块
        feature_lv3_1 = model3(images_lv3_1)
        feature_lv3_2 = model3(images_lv3_2)
        feature_lv3_3 = model3(images_lv3_3)
        feature_lv3_4 = model3(images_lv3_4)
        # 将第三尺度的进行合并
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        # 将第二尺度的图片与第三尺度融合在一起后合并
        feature_lv2_1 = model2(images_lv2_1 + feature_lv3_top)
        feature_lv2_2 = model2(images_lv2_2 + feature_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        # feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
        preds = model1(y + feature_lv2)
        # preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

if __name__ == '__main__':

    test_list = glob.glob("test_failuredehaze/*")
    for image in test_list:
        SRCNN_image(image)
        print(image, "done!")
