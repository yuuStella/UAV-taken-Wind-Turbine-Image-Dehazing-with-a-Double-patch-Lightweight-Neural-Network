import torch
import torchvision
import torch.optim
import time
import image_data_loader
import dehazeNet
import numpy as np
from PIL import Image
from torch.autograd import Variable
import glob


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    ld_net1 = dehazeNet.LightDehaze_Net().cuda()
    ld_net1.load_state_dict(torch.load('trained_weights/trained_LDNet122.pth'))

    ld_net2 = dehazeNet.LightDehaze_Net().cuda()
    ld_net2.load_state_dict(torch.load('trained_weights/trained_LDNet222.pth'))
    # clean_image = dehaze_net(data_hazy)
    H = data_hazy.size(2)
    W = data_hazy.size(3)

    images_lv1 = Variable(data_hazy - 0.5).cuda()

    images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
    images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

    feature_lv2_1 = ld_net2(images_lv2_1)
    feature_lv2_2 = ld_net2(images_lv2_2)

    feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + images_lv1

    dehaze_image = ld_net1(feature_lv2)
    torchvision.utils.save_image(dehaze_image, "results/" + image_path.split("/")[-1])


if __name__ == '__main__':

    test_list = glob.glob("test_images/*")
    time_sum = 0
    for image in test_list:
        time_start = time.time()
        dehaze_image(image)
        time_end = time.time()
        time_sum = time_end - time_start + time_sum
        print(time_sum)
        print(image, "done!")
    print(time_sum)