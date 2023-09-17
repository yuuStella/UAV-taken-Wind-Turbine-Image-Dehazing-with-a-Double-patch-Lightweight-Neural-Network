import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torchvision
from torch.autograd import Variable

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model1 = SRCNN().to(device)
    model2 = SRCNN().to(device)
    model3 = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam([
        {'params': model1.conv1.parameters()},
        {'params': model1.conv2.parameters()},
        {'params': model1.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    optimizer2 = optim.Adam([
        {'params': model2.conv1.parameters()},
        {'params': model2.conv2.parameters()},
        {'params': model2.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    optimizer3 = optim.Adam([
        {'params': model3.conv1.parameters()},
        {'params': model3.conv2.parameters()},
        {'params': model3.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights1 = copy.deepcopy(model1.state_dict())
    best_epoch1 = 0
    best_psnr1 = 0.0

    best_weights2 = copy.deepcopy(model2.state_dict())
    best_epoch2 = 0
    best_psnr2 = 0.0

    best_weights3 = copy.deepcopy(model3.state_dict())
    best_epoch3 = 0
    best_psnr3 = 0.0

    for epoch in range(args.num_epochs):
        model1.train()
        model2.train()
        model3.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                H = inputs.size(2)
                W = inputs.size(3)
                images_lv1 = Variable(inputs - 0.5).cuda()
                
                images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
                images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

                torchvision.utils.save_image(images_lv2_1, "./hazefree_edge/" + str(103) + ".jpg")
                
                images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
                images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
                images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
                images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]
                
                feature_lv3_1 = model3(images_lv3_1)
                feature_lv3_2 = model3(images_lv3_2)
                feature_lv3_3 = model3(images_lv3_3)
                feature_lv3_4 = model3(images_lv3_4)
                
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                
                feature_lv2_1 = model2(images_lv2_1 + feature_lv3_top)
                feature_lv2_2 = model2(images_lv2_2 + feature_lv3_bot)
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
                # feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                preds = model1(inputs + feature_lv2)

                # preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                optimizer2.step()

                optimizer3.zero_grad()
                optimizer3.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model1.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}1.pth'.format(epoch)))
        torch.save(model2.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}2.pth'.format(epoch)))
        torch.save(model3.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}3.pth'.format(epoch)))

        model1.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                H = inputs.size(2)
                W = inputs.size(3)
                images_lv1 = Variable(inputs - 0.5).cuda()
                
                images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
                images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

                torchvision.utils.save_image(images_lv2_1, "./hazefree_edge/" + str(103) + ".jpg")
                
                images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
                images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
                images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
                images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]
                
                feature_lv3_1 = model3(images_lv3_1)
                feature_lv3_2 = model3(images_lv3_2)
                feature_lv3_3 = model3(images_lv3_3)
                feature_lv3_4 = model3(images_lv3_4)
                
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                
                feature_lv2_1 = model2(images_lv2_1 + feature_lv3_top)
                feature_lv2_2 = model2(images_lv2_2 + feature_lv3_bot)
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
                
                preds = model1(inputs + feature_lv2).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr1:
            best_epoch1 = epoch
            best_psnr1 = epoch_psnr.avg
            best_weights1 = copy.deepcopy(model1.state_dict())
            best_epoch2 = epoch
            best_psnr2 = epoch_psnr.avg
            best_weights2 = copy.deepcopy(model2.state_dict())
            best_epoch3 = epoch
            best_psnr3 = epoch_psnr.avg
            best_weights3 = copy.deepcopy(model3.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch1, best_psnr1))
    torch.save(best_weights1, os.path.join(args.outputs_dir, 'best1.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch2, best_psnr2))
    torch.save(best_weights2, os.path.join(args.outputs_dir, 'best2.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch3, best_psnr3))
    torch.save(best_weights3, os.path.join(args.outputs_dir, 'best3.pth'))
