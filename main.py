#####################################################
#
# 改为WGAN-GP
#####################################################
import os
import sys
import socket
import pickle
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

from config import get_cfg
from dataset import generate_random_layout, wireframeDataset_Rplan
from models import Generator, WireframeDiscriminator, weight_init, renderer_g2v
from utils import bounds_check, get_figure, draw_table

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")

    # manually add attributes
    cfg.MANUAL.HOSTNAME = socket.gethostname()
    cfg.MANUAL.RPLAN_PATH = cfg.PATH.RPLAN[
        cfg.SYSTEM.HOSTNAMES.index(cfg.MANUAL.HOSTNAME)
    ]
    cfg.MANUAL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.MANUAL.DATE = datetime.now().strftime('%Y-%m-%d')
    cfg.MANUAL.TIME = datetime.now().strftime('%H-%M-%S')
    cfg.TENSORBOARD.ANNOTATION = os.popen("git log --pretty=oneline").read()\
        .split("\n")[0]\
        .replace(' ', '\n', 1)
    return cfg


def main(FLAGS):
    cfg = setup()
    device = torch.device(cfg.MANUAL.DEVICE)
    date = cfg.MANUAL.DATE
    time_ = cfg.MANUAL.TIME
    local_rank = FLAGS.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    print('using dataset:\t{}'.format(cfg.DATASET.NAME))
    print('hostname:\t{}'.format(cfg.MANUAL.HOSTNAME))
    print('date:\t\t{}'.format(cfg.MANUAL.DATE))
    # rooms=(0,1)
    #name_particular_rooms(path=path_rplan, rooms=rooms)
    # types_more_than_n(path=path_rplan,n=2000)

    real_dataset = wireframeDataset_Rplan(cfg)
    checkpoint = './params/params_rplan_{0}.pkl'.format(date)
    log_dir = cfg.PATH.LOG_DIR
    batch_size = cfg.DATASET.BATCHSIZE

    train_sampler = DistributedSampler(real_dataset)
    real_dataloader = DataLoader(
        real_dataset, batch_size, sampler=train_sampler,
        shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS, drop_last=True, pin_memory=True)

    # 固定的随机layout
    #fixed_z_file = './fixed_z/fixed_xyaw_{0}_1224.pkl'.format(dataset)
    fixed_z_file = cfg.PATH.Z_FILE
    if not os.path.exists(fixed_z_file):
        fixed = generate_random_layout(real_dataset, 8)
        #fixed_images = [torch.tensor(x).to(device) for x in fixed]
        with open(fixed_z_file, 'wb') as output:
            pickle.dump(fixed, output)
        print('generating'+fixed_z_file)

    # 初始化
    ######################################################
    learning_rate = cfg.TRAIN.LEARNING_RATE
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    ######################################################

    renderer = renderer_g2v(
        render_size=cfg.MODEL.RENDERER.RENDERING_SIZE, class_num=real_dataset.enc_len)

    generator = Generator(dataset=real_dataset).to(local_rank)
    generator.apply(weight_init)

    discriminator = WireframeDiscriminator(
        dataset=real_dataset, renderer=renderer).to(local_rank)
    discriminator.apply(weight_init)

    if dist.get_rank() == 0 and os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        epoch_last = checkpoint['epoch']
        n_iter = checkpoint['n_iter']
        print('load_previous_model')
    else:
        os.makedirs("./params", exist_ok=True)
        epoch_last = 0
        # global step
        n_iter = -1
        print('create_new_model')

    generator = DDP(generator)
    discriminator = DDP(discriminator)

    # Initialize optimizers.
    generator_optimizer = optim.Adam(generator.parameters(), learning_rate)
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), learning_rate)

    print('amount of parameters in generator:\t', sum(p.numel()
                                                      for p in generator.parameters() if p.requires_grad))
    print('amount of parameters in discriminator:\t', sum(p.numel()
                                                          for p in discriminator.parameters() if p.requires_grad))

    # 固定的随机噪声
    with open(fixed_z_file, 'rb') as pkl_file:
        fixed_z = pickle.load(pkl_file)
    fixed_z = [torch.tensor(x).to(device) for x in fixed_z]
    fig_fixed = get_figure(renderer.render(fixed_z[0].detach()))

    # 训练
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir+'/'+date+' '+time_)
        writer.add_text('annotation', cfg.TENSORBOARD.ANNOTATION, 1)
        writer.add_figure('fixed Z Image', fig_fixed, 1)

    # Start training.
    for ec in range(num_epochs):
        epoch = ec + epoch_last
        real_dataloader.sampler.set_epoch(epoch)
        print('Start to train epoch %d.' % (epoch))

        discriminator_losses = []
        generator_losses = []
        boundray_losses = []

        for batch_i, real_images in enumerate(real_dataloader, 0):
            n_iter += 1
            if n_iter % 10 == 9:
                print('\t iter {0} | batch {1} of epoch {2}'.format(
                    n_iter, batch_i, epoch))

            real_images = [x.to(device) for x in real_images]

            random_images = generate_random_layout(real_dataset, batch_size)
            random_images = [torch.tensor(x).to(device) for x in random_images]

            # 训练判别器
            generator.eval()
            discriminator.train()
            discriminator.zero_grad()
            pred_real = discriminator(real_images[0], real_images[1])
            pred_real.backward(torch.ones_like(pred_real).to(device))

            fake_images = generator(random_images[0], random_images[1])
            pred_fake = discriminator(
                fake_images[0].detach(), fake_images[1].detach())
            pred_fake.backward(-torch.ones_like(pred_real).to(device))

            discriminator_optimizer.step()
            discriminator_loss = pred_fake.mean() - pred_real.mean()
            discriminator_losses.append(
                discriminator_loss.cpu().detach().numpy())

            if (batch_i+1) % 2 == 0:
                # 训练生成器
                generator.train()
                discriminator.eval()
                generator.zero_grad()

                boundray_loss = bounds_check(fake_images[0])
                boundray_loss.backward(retain_graph=True)

                pred_fake = discriminator(fake_images[0], fake_images[1])

                pred_fake.backward(torch.ones_like(pred_real).to(device))
                generator_optimizer.step()
                generator_loss = pred_fake.mean()

                generator_losses.append(
                    generator_loss.cpu().detach().numpy())
                boundray_losses.append(
                    boundray_loss.cpu().detach().numpy())

        if dist.get_rank() == 0:
            # skip saving interval
            if epoch % cfg.TENSORBOARD.SAVE_INTERVAL_EPOCHS != cfg.TENSORBOARD.SAVE_INTERVAL_EPOCHS-1:
                continue
            # TensorboardX
            discriminator_losses = np.array(discriminator_losses)
            generator_losses = np.array(generator_losses)
            boundray_losses = np.array(boundray_losses)
            #gradient_penalties = np.array(gradient_penalties)

            print('N_iter {:7d} | Epoch [{:5d}/{:5d}] | discriminator_loss: {:6.4f} | generator_loss: {:6.4f}'.
                  format(n_iter, epoch, num_epochs, discriminator_losses.mean(), generator_losses.mean()))
            generator.eval()
            discriminator.eval()

            # 记录梯度
            try:
                G_grads = [(name, weight.grad.mean(
                )) for name, weight in generator.named_parameters() if weight.requires_grad]
                writer.add_scalar('G_grads 0 {}'.format(G_grads[0][0]),
                                  G_grads[0][1], n_iter)
                writer.add_scalar('G_grads -1 {}'.format(G_grads[-1][0]),
                                  G_grads[-1][1], n_iter)
            except:
                print('G_layer_zero_grad')

            try:
                D_grads = [(name, weight.grad.mean(
                )) for name, weight in discriminator.named_parameters() if weight.requires_grad]
                writer.add_scalar('D_grads 0 {}'.format(D_grads[0][0]),
                                  D_grads[0][1], n_iter)
                writer.add_scalar('D_grads -1 {}'.format(D_grads[-1][0]),
                                  D_grads[-1][1], n_iter)
            except:
                print('D_layer_zero_grad')

            # 记录loss
            writer.add_scalar('Discriminator Loss',
                              discriminator_losses.mean(), n_iter)
            writer.add_scalar('Generator Loss',
                              generator_losses.mean(), n_iter)
            writer.add_scalar(
                'Boundray Loss', boundray_losses.mean(), n_iter)
            #writer.add_scalar('Gradient_penalties', gradient_penalties.mean(), n_iter)

            # 记录网络权重
            for name, param in discriminator.named_parameters():
                writer.add_histogram(
                    'Discriminator: ' + name, param.clone().detach().cpu().data.numpy(), n_iter)
            for name, param in generator.named_parameters():
                writer.add_histogram(
                    'Generator: ' + name, param.clone().detach().cpu().data.numpy(), n_iter)

            # 每个epoch向tensorboard添加recall
            real_pred = pred_real.detach().cpu().numpy()
            random_pred = discriminator(
                random_images[0], random_images[1]).detach().cpu().numpy()
            fake_pred = pred_fake.detach().cpu().numpy()

            acc = plt.figure(figsize=(10, 5))
            ax = acc.add_subplot(111)
            ax.hist(real_pred, alpha=0.5, density=True, bins=60)
            ax.hist(random_pred, alpha=0.5, density=True, bins=60)
            ax.hist(fake_pred, alpha=0.5, density=True, bins=60)
            ax.legend(['real_pred', 'random_pred', 'fake_pred'])
            writer.add_figure('prediction', acc, n_iter)

            # 可视化真实图像,添加到tensorboard
            fig_real = get_figure(renderer.render(real_images[0][:8]))
            writer.add_figure('Real Image', fig_real, n_iter)

            # 可视化Z生成图像
            generated_z = generator(fixed_z[0], fixed_z[1])
            fig_fake = get_figure(renderer.render(generated_z[0]))
            writer.add_figure('Fake Image', fig_fake, n_iter)

            table = draw_table(generated_z[0][4])
            writer.add_figure('generated[4]', table, n_iter)

            # 在确认模型有效之前，暂不保存模型参数
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
            }, checkpoint)
            print('\tparams saved to ' + checkpoint)

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    FLAGS = parser.parse_args()
    main(FLAGS)
