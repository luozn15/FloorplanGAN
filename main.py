#####################################################
#
# 改为WGAN-GP
#####################################################
import os
import sys
import socket
import pickle
from threading import local
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from config import get_cfg
from dataset import generate_random_layout, wireframeDataset_Rplan
from models import Generator, WireframeDiscriminator, weight_init, renderer_g2v
from utils import bounds_check, get_figure, draw_table, negative_wh_check

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def real_loss(D_out, smooth=False):
    if smooth:
        labels = torch.ones_like(D_out) * (np.random.rand()*(1-smooth)+smooth)
    else:
        labels = torch.ones_like(D_out)
    #crit = nn.BCEWithLogitsLoss()
    crit = nn.BCELoss()
    loss = crit(D_out, labels)
    return loss


def fake_loss(D_out, smooth=False):
    if smooth:
        labels = torch.ones_like(D_out) * (np.random.rand()*smooth)
    else:
        labels = torch.zeros_like(D_out)
    #crit = nn.BCEWithLogitsLoss()
    crit = nn.BCELoss()
    loss = crit(D_out, labels)
    return loss


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")

    # manually add attributes
    cfg.MANUAL.HOSTNAME = socket.gethostname()
    if cfg.MANUAL.HOSTNAME in cfg.SYSTEM.HOSTNAMES:
        cfg.MANUAL.RPLAN_PATH = cfg.PATH.RPLAN[
            cfg.SYSTEM.HOSTNAMES.index(cfg.MANUAL.HOSTNAME)
        ]
    else:
        cfg.MANUAL.RPLAN_PATH = "./data_FloorplanGAN/pkls"
    #cfg.MANUAL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.MANUAL.DATE = datetime.now().strftime('%Y-%m-%d')
    cfg.MANUAL.TIME = datetime.now().strftime('%H-%M-%S')
    cfg.TENSORBOARD.ANNOTATION = os.popen("git log --pretty=oneline").read()\
        .split("\n")[0]\
        .replace(' ', '\n', 1)
    cfg.freeze()
    os.makedirs(cfg.PATH.LOG_DIR+'/'+cfg.MANUAL.DATE +
                ' '+cfg.MANUAL.TIME, exist_ok=True)
    with open(cfg.PATH.LOG_DIR+'/'+cfg.MANUAL.DATE+' '+cfg.MANUAL.TIME+'/'+'config.yaml', "w") as f:
        f.write(cfg.dump())
    return cfg


def train(parallel=False):
    cfg = setup()

    print('dataset:\t{}'.format(cfg.DATASET.NAME))
    print('hostname:\t{}'.format(cfg.MANUAL.HOSTNAME))
    print('date:\t\t{}'.format(cfg.MANUAL.DATE))

    if parallel:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0")
    real_dataset = wireframeDataset_Rplan(cfg)
    checkpoint = './params/params_rplan_{0}.pkl'.format(cfg.MANUAL.DATE)
    batch_size = cfg.DATASET.BATCHSIZE

    if parallel:
        ddp_sampler = DistributedSampler(real_dataset)
        real_dataloader = DataLoader(
            real_dataset, batch_size, sampler=ddp_sampler,
            num_workers=cfg.SYSTEM.NUM_WORKERS, drop_last=True, pin_memory=True)
    else:
        real_dataloader = DataLoader(
            real_dataset, batch_size, shuffle=True,
            num_workers=cfg.SYSTEM.NUM_WORKERS, drop_last=True, pin_memory=True)

    # 固定的随机layout
    fixed_z_file = cfg.PATH.Z_FILE

    # 初始化
    ######################################################
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    ######################################################

    renderer = renderer_g2v(
        render_size=cfg.MODEL.RENDERER.RENDERING_SIZE, class_num=real_dataset.enc_len)

    generator = Generator(dataset=real_dataset).to(device)
    # generator.apply(weight_init)
    discriminator = WireframeDiscriminator(
        dataset=real_dataset, renderer=renderer, cfg=cfg).to(device)
    # discriminator.apply(weight_init)

    if os.path.exists(checkpoint):
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
    if parallel:
        generator = DDP(generator, device_ids=[
                        local_rank], output_device=local_rank, broadcast_buffers=False)
        discriminator = DDP(discriminator, device_ids=[
                            local_rank], output_device=local_rank, broadcast_buffers=False)

    # Initialize optimizers.
    generator_optimizer = optim.Adam(
        generator.parameters(), cfg.MODEL.GENERATOR.LEARNING_RATE)
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), cfg.MODEL.DISCRIMINATOR.LEARNING_RATE)
    """generator_scheduler = optim.lr_scheduler.ExponentialLR(
        generator_optimizer, 0.9)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(
        discriminator_optimizer, 0.9)"""
    epoch_step = len(real_dataset) // batch_size
    generator_scheduler = optim.lr_scheduler.StepLR(
        generator_optimizer, step_size=cfg.MODEL.GENERATOR.DECAY_EPOCHS*epoch_step, gamma=0.9)  # 每30 epoch decay lr
    discriminator_scheduler = optim.lr_scheduler.StepLR(
        discriminator_optimizer, step_size=cfg.MODEL.DISCRIMINATOR.DECAY_EPOCHS*epoch_step, gamma=0.9)

    if not parallel or dist.get_rank() == 0:
        print('amount of parameters in generator:\t',
              sum(p.numel() for p in generator.parameters() if p.requires_grad))
        print('amount of parameters in discriminator:\t',
              sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

        # 固定的随机噪声
        if not os.path.exists(fixed_z_file):
            fixed = generate_random_layout(real_dataset, 8)
            #fixed_images = [torch.tensor(x).to(device) for x in fixed]
            with open(fixed_z_file, 'wb') as output:
                pickle.dump(fixed, output)
            print('generating'+fixed_z_file)
        with open(fixed_z_file, 'rb') as pkl_file:
            fixed_z = pickle.load(pkl_file)
        fixed_z = [torch.tensor(x).to(device) for x in fixed_z]
        fig_fixed = get_figure(renderer.render(fixed_z[0].detach()))

        # 训练
        writer = SummaryWriter(log_dir=cfg.PATH.LOG_DIR +
                               '/'+cfg.MANUAL.DATE+' '+cfg.MANUAL.TIME)
        writer.add_text('annotation', cfg.TENSORBOARD.ANNOTATION, 1)
        writer.add_figure('fixed Z Image', fig_fixed, 1)

    # Start training.
    EPSILON = 10e-8
    for ec in range(num_epochs):
        epoch = ec + epoch_last
        print('Start to train epoch %d.' % (epoch))
        if parallel:
            real_dataloader.sampler.set_epoch(epoch)
        discriminator_losses = []
        discriminator_losses_real = []
        discriminator_losses_fake = []
        generator_losses = []
        boundray_losses = []
        negative_wh_losses = []

        for batch_i, real_images in enumerate(real_dataloader, 0):
            n_iter += 1
            generator.train()
            discriminator.train()

            real_images = [x.to(device) for x in real_images]

            random_images = generate_random_layout(real_dataset, batch_size)
            random_images = [torch.tensor(x).to(device) for x in random_images]

            if batch_i % 2 == 0:
                # 训练判别器
                # generator.eval()
                # discriminator.train()
                discriminator_optimizer.zero_grad()

                pred_real = discriminator(real_images[0], real_images[1])
                d_real_loss = real_loss(pred_real, smooth=0.9)
                #w_real = 1./(pred_real.mean().item()+EPSILON)
                #w_real = (1.-pred_real)**2

                fake_images = generator(random_images[0], random_images[1])
                pred_fake = discriminator(
                    fake_images[0].detach(), fake_images[1].detach())
                d_fake_loss = fake_loss(pred_fake, smooth=0.1)
                #w_fake = 1./(1.-pred_fake.mean().item()+EPSILON)
                #w_fake = pred_fake**2

                #d_loss = ((w_real * d_real_loss + w_fake * d_fake_loss)/(w_real+w_fake)).mean()
                d_loss = d_real_loss+d_fake_loss
                #d_loss = d_fake_loss
                d_loss.backward()
                discriminator_optimizer.step()
                discriminator_scheduler.step()
                #print("D: pred_real {:6.4f} pred_fake {:6.4f}".format(pred_real.mean().item(), pred_fake.mean().item()))
                discriminator_losses_real.append(
                    d_real_loss.cpu().detach().numpy())
                discriminator_losses_fake.append(
                    d_fake_loss.cpu().detach().numpy())
                discriminator_losses.append(d_loss.cpu().detach().numpy())

            if batch_i % 2 == 0:
                generator_optimizer.zero_grad()
            if batch_i % 1 == 0:
                # 训练生成器
                # generator.train()
                # discriminator.eval()

                fake_images = generator(random_images[0], random_images[1])
                boundray_loss = bounds_check(fake_images[0])
                negative_wh_loss = negative_wh_check(fake_images[0])

                pred_fake = discriminator(fake_images[0], fake_images[1])
                g_loss = real_loss(pred_fake, False)
                sum_loss = g_loss + boundray_loss + negative_wh_loss
                sum_loss.backward()
            if batch_i % 2 == 0:
                generator_optimizer.step()
                generator_scheduler.step()
                #print("G: pred_fake {:6.4f}".format(pred_fake.mean().item()))
                generator_losses.append(
                    g_loss.cpu().detach().numpy())
                boundray_losses.append(boundray_loss.cpu().detach().numpy())
                negative_wh_losses.append(
                    negative_wh_loss.cpu().detach().numpy())

            if n_iter % 10 == 9:
                if parallel:
                    print('\t local_rank {} | iter {} | batch {} of epoch {} | G_loss {:6.4f} | D_loss {:6.4f}'
                          .format(dist.get_rank(), n_iter, batch_i, epoch, g_loss.item(), d_loss.item()))
                else:
                    print('\t iter {} | batch {} of epoch {} | G_loss {:6.4f} | D_loss {:6.4f}'
                          .format(n_iter, batch_i, epoch, g_loss.item(), d_loss.item()))

        """if epoch % cfg.MODEL.GENERATOR.DECAY_EPOCHS == cfg.MODEL.GENERATOR.DECAY_EPOCHS - 1:
            generator_scheduler.step()
        if epoch % cfg.MODEL.DISCRIMINATOR.DECAY_EPOCHS == cfg.MODEL.DISCRIMINATOR.DECAY_EPOCHS - 1:
            discriminator_scheduler.step()"""
        # skip saving interval
        if epoch % cfg.TENSORBOARD.SAVE_INTERVAL_EPOCHS != 0:
            continue
        # TensorboardX
        if not parallel or dist.get_rank() == 0:
            tensorboard_write(
                writer,
                n_iter, epoch, num_epochs,
                boundray_losses, negative_wh_losses,
                discriminator_losses_real, discriminator_losses_fake, discriminator_losses, generator_losses,
                generator, discriminator, renderer,
                discriminator_optimizer, generator_optimizer,
                pred_real, pred_fake, real_images, random_images, fixed_z
            )

            # 保存模型参数

            torch.save({
                'generator_state_dict': generator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
            }, "{}.pkl".format(checkpoint.split(".pkl")[0]))
            print('\tparams saved to ' + "{}.pkl".format(checkpoint.split(".pkl")[0]))

    writer.close()


def tensorboard_write(
    writer,
    n_iter, epoch, num_epochs,
    boundray_losses, negative_wh_losses,
    discriminator_losses_real, discriminator_losses_fake, discriminator_losses, generator_losses,
    generator, discriminator, renderer,
    discriminator_optimizer, generator_optimizer,
    pred_real, pred_fake, real_images, random_images, fixed_z
):
    discriminator_losses_real = np.array(discriminator_losses_real)
    discriminator_losses_fake = np.array(discriminator_losses_fake)
    discriminator_losses = np.array(discriminator_losses)
    generator_losses = np.array(generator_losses)
    boundray_losses = np.array(boundray_losses)
    negative_wh_losses = np.array(negative_wh_losses)

    print('N_iter {:7d} | Epoch [{:5d}/{:5d}] | discriminator_loss: {:6.4f} | generator_loss: {:6.4f}'.
          format(n_iter, epoch, num_epochs, discriminator_losses.mean(), generator_losses.mean()))
    # generator.eval()
    # discriminator.eval()

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
    writer.add_scalar(
        'Discriminator Loss Real', discriminator_losses_real.mean(), n_iter)
    writer.add_scalar(
        'Discriminator Loss Fake', discriminator_losses_fake.mean(), n_iter)
    writer.add_scalar(
        'Discriminator Loss', discriminator_losses.mean(), n_iter)
    writer.add_scalar(
        'Generator Loss', generator_losses.mean(), n_iter)
    writer.add_scalar(
        'Boundray Loss', boundray_losses.mean(), n_iter)
    writer.add_scalar(
        'Negative WH Loss', negative_wh_losses.mean(), n_iter)
    #writer.add_scalar('Gradient_penalties', gradient_penalties.mean(), n_iter)

    # 记录学习率
    writer.add_scalar(
        'Discriminator Learning Rate', discriminator_optimizer.state_dict()['param_groups'][0]['lr'], n_iter)
    writer.add_scalar(
        'Generator Learning Rate', generator_optimizer.state_dict()['param_groups'][0]['lr'], n_iter)

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

    acc = plt.figure(figsize=(8, 5))
    ax = acc.gca()
    ax.hist(real_pred, alpha=0.5, density=True,
            bins=60, range=(0, 1), label='real')
    ax.hist(random_pred, alpha=0.5, density=True,
            bins=60, range=(0, 1), label='random')
    ax.hist(fake_pred, alpha=0.5, density=True,
            bins=60, range=(0, 1), label='fake')
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # torchrun --nnodes=1 --nproc_per_node=2 main.py
    #import os
    #local_rank = int(os.environ["LOCAL_RANK"])

    # python -m torch.distributed.launch main.py

    if torch.cuda.device_count() > 1:
        # torchrun --nnodes=1 --nproc_per_node=2 main.py
        #import os
        #local_rank = int(os.environ["LOCAL_RANK"])

        # python -m torch.distributed.launch main.py
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        local_rank = args.local_rank
        print(local_rank)
        with torch.cuda.device(local_rank):
            torch.distributed.init_process_group(backend='nccl')
            train(parallel=True)
    else:
        # CUDA_VISIBLE_DEVICES=0 python main.py
        train(parallel=False)
