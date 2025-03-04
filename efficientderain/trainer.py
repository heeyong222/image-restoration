import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import encoding
from torchvision import transforms

import pytorch_ssim as pytorch_ssim
import dataset as dataset
import utils as utils


from torch.utils.tensorboard import SummaryWriter

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    cudnn.benchmark = opt.cudnn_benchmark

    save_folder = opt.save_path
    sample_folder = opt.save_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    if not opt.no_gpu:
        criterion_L1 = torch.nn.L1Loss().cuda()
        criterion_L2 = torch.nn.MSELoss().cuda()
        criterion_ssim = pytorch_ssim.SSIM().cuda()
    else: 
        criterion_L1 = torch.nn.L1Loss()
        criterion_L2 = torch.nn.MSELoss()
        criterion_ssim = pytorch_ssim.SSIM()

    # Initialize Generator
    generator = utils.create_generator(opt)

    if not opt.no_gpu:
        if opt.multi_gpu:
            generator = nn.DataParallel(generator)
            generator = generator.cuda()
        else:
            generator = generator.cuda()

    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                   lr=opt.lr_g, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    print("pretrained models loaded")

    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def save_model(opt, epoch, iteration, len_dataset, generator):
        if opt.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        elif opt.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % epoch)
            elif opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % iteration)
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % epoch)
            elif opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % iteration)

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    trainset = dataset.DenoisingDataset(opt)
    valset = dataset.DenoisingValDataset(opt)
    print('The overall number of training images:', len(trainset))

    train_loader = DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=8, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)

    # SummaryWriter 초기화
    writer = SummaryWriter(os.path.join(opt.save_path, 'logs'))

    prev_time = time.time()

    # 에폭별 손실 기록을 위한 리스트 (원한다면)
    train_losses = []
    learning_rates = []

    for epoch in range(opt.epochs):
        generator.train()
        epoch_loss = 0.0  # 에폭 손실 누적 변수

        for i, (true_input, true_target) in enumerate(train_loader):
            if not opt.no_gpu:
                true_input = true_input.cuda()
                true_target = true_target.cuda()

            optimizer_G.zero_grad()
            fake_target = generator(true_input, true_input)

            ssim_loss = -criterion_ssim(true_target, fake_target)
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            loss = Pixellevel_L1_Loss + 0.2 * ssim_loss
            loss.backward()
            optimizer_G.step()

            epoch_loss += loss.item()

            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f] Time_left: %s" %
                  (epoch + 1, opt.epochs, i, len(train_loader),
                   Pixellevel_L1_Loss.item(), ssim_loss.item(), time_left), end='')

            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        # 에폭 평균 손실 계산 및 기록
        avg_loss = epoch_loss / len(train_loader)
        print("\nEpoch %d Average Loss: %.4f" % (epoch + 1, avg_loss))
        train_losses.append(avg_loss)
        learning_rates.append(optimizer_G.param_groups[0]['lr'])
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)

        ### 에폭 종료 후 Validation 수행
        generator.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        num_val_images = 0

        with torch.no_grad():
            for (val_input, val_target) in val_loader:
                if not opt.no_gpu:
                    val_input = val_input.cuda()
                    val_target = val_target.cuda()
                pred = generator(val_input, val_input)
                batch_size = val_input.size(0)
                num_val_images += batch_size

                batch_psnr = utils.psnr(pred, val_target) * batch_size
                val_psnr_sum += batch_psnr

                batch_ssim = criterion_ssim(pred, val_target).item() * batch_size
                val_ssim_sum += batch_ssim

        avg_psnr = val_psnr_sum / num_val_images
        avg_ssim = val_ssim_sum / num_val_images
        print("Validation Epoch %d: PSNR: %.4f, SSIM: %.4f" % (epoch + 1, avg_psnr, avg_ssim))
        writer.add_scalar('PSNR/val', avg_psnr, epoch + 1)
        writer.add_scalar('SSIM/val', avg_ssim, epoch + 1)

        metrics_path = os.path.join(sample_folder, "val_metrics.txt")
        with open(metrics_path, "a") as f:
            f.write("Epoch %d: PSNR: %.4f, SSIM: %.4f\n" % (epoch + 1, avg_psnr, avg_ssim))

        ### 에폭 종료 후 체크포인트 저장 (pth 파일)
        checkpoint_filename = 'checkpoint_epoch%d.pth' % (epoch + 1)
        checkpoint_path = os.path.join(opt.save_path, checkpoint_filename)
        if opt.multi_gpu:
            torch.save(generator.module.state_dict(), checkpoint_path)
        else:
            torch.save(generator.state_dict(), checkpoint_path)
        print('Checkpoint saved at %s' % checkpoint_path)

        ### 매 에폭마다 Validation 데이터셋에서 샘플 이미지 저장 (Center Crop한 Validation 샘플)
        with torch.no_grad():
            for (sample_input, sample_target) in val_loader:
                if not opt.no_gpu:
                    sample_input = sample_input.cuda()
                    sample_target = sample_target.cuda()
                fake_sample = generator(sample_input, sample_input)
                break
        sample_img_list = [sample_input, fake_sample, sample_target]
        # 파일명을 고정하여 덮어쓰도록 할 수도 있습니다.
        sample_name = 'val_sample' + f"_{epoch + 1}"
        utils.save_sample_png(sample_folder=sample_folder,
                              sample_name=sample_name,
                              img_list=sample_img_list,
                              name_list=['in', 'pred', 'gt'],
                              pixel_max_cnt=255)

        generator.train()  # 학습 모드로 전환

    writer.close()