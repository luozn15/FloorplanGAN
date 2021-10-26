#####################################################
#
# 改为WGAN-GP
#####################################################
import scipy
import os
import sys
import socket
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm

from dataset import generate_random_layout, wireframeDataset_Rplan
from models import Generator_branch, Discriminator_visual, weight_init, renderer_g2v
from utils import bounds_check, get_figure, draw_table

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def main(argv):
    #########################################################################################################
    dataset_name = 'rplan'
    rooms = None
    subset_name = 'names_0-1_1-1_2-1_3-1_4-0_5-0_6-0_7-1_8-0_9-1.pkl'
    #fixed_z_file = './fixed_z/fixed_xyaw_{0}_0320.pkl'.format(dataset_name)
    fixed_z_file = './fixed_z/fixed_xywh_{0}_{1}'.format(
        dataset_name, subset_name)
    annotation = '''
        2021-04-27
        房间数固定
    '''
##########################################################################################################
    hostname = socket.gethostname()
    if hostname == 'ubuntuStation':
        path = '../../data_FloorPlan'
        path_rplan = '../../data_RPLAN/pkls'
    elif hostname == 'DESKTOP-HRFSC59':
        path = 'D:\\luozn\\data_FloorPlan'
        path_rplan = 'D:\\luozn\\data_RPLAN\\pkls'
        os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    elif hostname == 'Work-Station':
        path = 'F:\\luozn\\data_FloorPlan'
        path_rplan = 'F:\\luozn\\data_RPLAN\\pkls'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    elif hostname == 'LAPTOP-LUOZN':
        path = 'E:\Seafile\data_FloorPlan'
        path_rplan = 'E:\\Seafile\\data_RPLAN\\pkls'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    date = datetime.now().strftime('%Y-%m-%d')
    time_ = datetime.now().strftime('%H:%M:%S')
    #dataset = 'floorplan'
    print('using dataset:\t{}'.format(dataset_name))
    print('hostname:\t{}'.format(hostname))
    print('date:\t\t{}'.format(date))
    # rooms=(0,1)
    #name_particular_rooms(path=path_rplan, rooms=rooms)
    # types_more_than_n(path=path_rplan,n=2000)

    if dataset_name == 'floorplan':
        real_dataset = wireframeDataset_Rplan(path=path)
        pkl_name = './params/params_floorplan_{0}.pkl'.format(date)
        batch_size = 16
        log_dir = 'runs_floorplan'
    elif dataset_name == 'rplan':
        real_dataset = wireframeDataset_Rplan(
            path=path_rplan, subset_name=subset_name)
        pkl_name = './params/params_rplan_{0}.pkl'.format(date)
        batch_size = 128
        log_dir = 'runs_rplan'
    real_dataloader = DataLoader(
        real_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)

    # 固定的随机layout
    #fixed_z_file = './fixed_z/fixed_xyaw_{0}_1224.pkl'.format(dataset)
    if not os.path.exists(fixed_z_file):
        fixed = generate_random_layout(real_dataset, 8)
        #fixed_images = [torch.tensor(x).to(device) for x in fixed]
        with open(fixed_z_file, 'wb') as output:
            pickle.dump(fixed, output)
        print('generating'+fixed_z_file)

    # 初始化
    ######################################################
    learning_rate = 0.00002
    clamp_num = 0.01
    num_epochs = 3000
    tensorboard_interval = 100
    tensorboard = True
    ######################################################
    discriminator_losses = []
    generator_losses = []
    boundray_losses = []
    gradient_penalties = []

    renderer = renderer_g2v(render_size=64, class_num=real_dataset.enc_len)
    def render(x): return render_template(x, 64, real_dataset.enc_len)

    generator = Generator_branch(dataset=real_dataset)
    generator.apply(weight_init)
    generator = nn.DataParallel(generator)
    generator.to(device)
    discriminator = Discriminator_visual(
        dataset=real_dataset, renderer=renderer)
    discriminator.apply(weight_init)
    discriminator = nn.DataParallel(discriminator)
    discriminator.to(device)

    # Initialize optimizers.
    generator_optimizer = optim.RMSprop(generator.parameters(), learning_rate)
    discriminator_optimizer = optim.RMSprop(
        discriminator.parameters(), learning_rate)  # Wasserstein GAN推荐使用RMSProp优化

    if os.path.exists(pkl_name):
        checkpoint = torch.load(pkl_name)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(
            checkpoint['generator_optimizer_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator_optimizer.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict'])
        epoch_last = checkpoint['epoch']
        n_iter = checkpoint['n_iter']
        print('load_previous_model')
    else:
        epoch_last = 0
        # global step
        n_iter = -1
        print('create_new_model')

    print('amount of parameters in generator:\t', sum(p.numel()
                                                      for p in generator.parameters() if p.requires_grad))
    print('amount of parameters in discriminator:\t', sum(p.numel()
                                                          for p in discriminator.parameters() if p.requires_grad))

    # 训练
    writer = SummaryWriter(log_dir=log_dir+'/'+date+' '+time_)
    writer.add_text('annotation', annotation, 1)

    # 固定的随机噪声
    with open(fixed_z_file, 'rb') as pkl_file:
        fixed_z = pickle.load(pkl_file)
    fixed_z = [torch.tensor(x).to(device) for x in fixed_z]
    discriminator.eval()
    fig_fixed = get_figure(render(fixed_z[0].detach()))
    writer.add_figure('fixed Z Image', fig_fixed, 1)

    # Start training.
    for ec in range(num_epochs):
        epoch = ec + epoch_last
        print('Start to train epoch %d.' % (epoch))

        for batch_i, real_images in enumerate(real_dataloader, 0):
            n_iter += 1

            print('\t iter {0} | batch {1} of epoch {2}'.format(
                n_iter, batch_i, epoch))

            discriminator.train()
            generator.train()

            '''real_images = [torch.cat([F.relu(torch.normal(mean=real_images[0][:,:,:-4],std=0.02)),\
                                      real_images[0][:,:,-4:]],axis=-1),\
                           real_images[1]]#向真实数据加噪声'''
            real_images = [x.to(device) for x in real_images]

            random_images = generate_random_layout(real_dataset, batch_size)
            random_images = [torch.tensor(x).to(device) for x in random_images]

            # 训练判别器
            discriminator.zero_grad()
            pred_real = discriminator(real_images[0], real_images[1])

            fake_images = generator(random_images[0], random_images[1])
            pred_fake = discriminator(
                fake_images[0].detach(), fake_images[1].detach())

            # Gradient Penalty
            alpha = torch.rand(batch_size, 1, 1)
            alpha = alpha.expand(real_images[0].size())
            alpha = alpha.to(device)
            interpolates = alpha * \
                real_images[0] + (1 - alpha) * fake_images[0].detach()
            interpolates = torch.autograd.Variable(
                interpolates, requires_grad=True)
            pred_interpolates = discriminator(interpolates, real_images[1])
            gradients = torch.autograd.grad(outputs=pred_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(
                                                pred_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                                ** 2).mean() * 10  # * LAMBDA

            # D_loss
            discriminator_loss = -pred_real.mean() + pred_fake.mean() + gradient_penalty

            discriminator_loss.backward()
            discriminator_optimizer.step()

            if tensorboard:
                discriminator_losses.append(
                    (pred_real.mean()-pred_fake.mean()).cpu().detach().numpy())
                gradient_penalties.append(
                    gradient_penalty.cpu().detach().numpy())

            if (batch_i+1) % 5 == 0:
                # 训练生成器
                generator.zero_grad()

                boundray_loss = bounds_check(fake_images[0])
                boundray_loss.backward(retain_graph=True)

                pred_fake = discriminator(fake_images[0], fake_images[1])
                generator_loss = -pred_fake.mean()

                generator_loss.backward()
                generator_optimizer.step()

                if tensorboard:
                    generator_losses.append(
                        generator_loss.cpu().detach().numpy())
                    boundray_losses.append(
                        boundray_loss.cpu().detach().numpy())

            # TensorboardX
            if n_iter % tensorboard_interval == tensorboard_interval-1:
                discriminator_losses = np.array(discriminator_losses)
                generator_losses = np.array(generator_losses)
                boundray_losses = np.array(boundray_losses)
                gradient_penalties = np.array(gradient_penalties)

                print('\t\t N_iter {:7d} | Epoch [{:5d}/{:5d}] | -discriminator_loss: {:6.4f} | generator_loss: {:6.4f}'.
                      format(n_iter, epoch, num_epochs, discriminator_losses.mean(), generator_losses.mean()))
                if tensorboard:
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
                    writer.add_scalar('-Discriminator Loss',
                                      discriminator_losses.mean(), n_iter)
                    writer.add_scalar('Generator Loss',
                                      generator_losses.mean(), n_iter)
                    writer.add_scalar(
                        'Boundray Loss', boundray_losses.mean(), n_iter)
                    writer.add_scalar('Gradient_penalties',
                                      gradient_penalties.mean(), n_iter)

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
                    fig_real = get_figure(render(real_images[0][:8]))
                    writer.add_figure('Real Image', fig_real, n_iter)

                    # 可视化Z生成图像
                    generated_z = generator(fixed_z[0], fixed_z[1])
                    fig_fake = get_figure(render(generated_z[0]))
                    writer.add_figure('Fake Image', fig_fake, n_iter)

                    table = draw_table(generated_z[0][4])
                    writer.add_figure('generated[4]', table, n_iter)

                discriminator_losses = []
                generator_losses = []
                boundray_losses = []
                gradient_penalties = []

        # 在确认模型有效之前，暂不保存模型参数
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'epoch': epoch,
            'n_iter': n_iter,
        }, pkl_name)
        print('\tparams saved to ' + pkl_name)

    if tensorboard:
        writer.close()


if __name__ == '__main__':
    main(sys.argv)
