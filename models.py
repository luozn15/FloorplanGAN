import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

# Stacked relation module


class RelationNonLocal(nn.Module):
    def __init__(self, C):
        super(RelationNonLocal, self).__init__()
        self.conv_fv = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fk = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fq = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.conv_fr = nn.Conv2d(C, C, kernel_size=1, stride=1)

    def forward(self, input_):
        N, C, H, W = input_.shape
        f_v = self.conv_fv(input_)  # (N, C, H, W)
        f_k = self.conv_fk(input_)
        f_q = self.conv_fq(input_)

        f_k = f_k.reshape([N, C, H*W]).permute(0, 2, 1)  # (N,H*W,C)
        f_q = f_q.reshape([N, C, H*W])
        w = torch.matmul(f_k, f_q)/(H*W)  # (N,H*W,H*W)

        f_r = torch.matmul(w.permute(0, 2, 1), f_v.reshape(
            [N, C, H*W]).permute(0, 2, 1)).permute(0, 2, 1)
        f_r = f_r.reshape(N, C, H, W)
        f_r = self.conv_fr(f_r)
        return f_r


class Generator(nn.Module):
    def __init__(self, dataset):  # feature_size, class_num, element_num):
        super(Generator, self).__init__()

        self.class_num = dataset.enc_len  # 6
        self.element_num = dataset.maximum_elements_num  # 6
        dim = 64
        # Encoder: two fully connected layers, input layout Z.
        self.encoder_fc0 = nn.Conv2d(
            4, dim*4, kernel_size=1, stride=1, bias=False)  # 4 是 (xc,w,yc,h) 的位数
        self.encoder_batch_norm0 = nn.BatchNorm2d(dim*4)
        self.encoder_fc1 = nn.Conv2d(
            4, dim, kernel_size=1, stride=1, bias=False)
        self.encoder_batch_norm1 = nn.BatchNorm2d(dim)
        self.encoder_fc2 = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=False)
        self.encoder_batch_norm2 = nn.BatchNorm2d(dim)
        self.encoder_fc3 = nn.Conv2d(
            dim, dim*4, kernel_size=1, stride=1, bias=False)
        self.encoder_batch_norm3 = nn.BatchNorm2d(dim*4)

        # Batchsize=16, Chanel=dim*4 ,S=8, O=10
        """encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim*4*self.class_num, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3)"""
        #
        self.relation_nonLocal0 = RelationNonLocal(
            256*self.class_num)
        self.relation_bn0 = nn.BatchNorm2d(256*self.class_num)
        self.relation_nonLocal1 = RelationNonLocal(
            256*self.class_num)
        self.relation_bn1 = nn.BatchNorm2d(256*self.class_num)

        self.relation_nonLocal2 = RelationNonLocal(
            256*self.class_num)
        self.relation_bn2 = nn.BatchNorm2d(256*self.class_num)
        self.relation_nonLocal3 = RelationNonLocal(
            256*self.class_num)
        self.relation_bn3 = nn.BatchNorm2d(256*self.class_num)

        # Decoder, two fully connected layers.
        self.decoder_fc0 = nn.Conv2d(
            dim*4*self.class_num, dim*4, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm0 = nn.BatchNorm2d(dim*4)
        self.decoder_fc1 = nn.Conv2d(
            dim*4*self.class_num, dim, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm1 = nn.BatchNorm2d(dim)
        self.decoder_fc2 = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm2 = nn.BatchNorm2d(dim)
        self.decoder_fc3 = nn.Conv2d(
            dim, dim*4, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm3 = nn.BatchNorm2d(dim*4)

        # Branch of class
        # self.branch_fc0 = nn.Conv2d(1024, self.class_num,kernel_size=1,stride=1) #(16,10,8,1)
        #self.sigmoid_brach0 = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)
        # Branch of position
        self.branch_fc1 = nn.Conv2d(
            dim*4, 4, kernel_size=1, stride=1)  # (16,4,8,1)
        # self.branch_fc1 = nn.Linear(self.class_num, 1)  # (16,4,8,1)
        #self.sigmoid_brach1 = nn.Sigmoid()
        for param in self.named_parameters():
            torch.nn.init.normal_(param[1].data, mean=0.0, std=0.02)

    def forward(self, input_data, input_length):  # (B,S,14),(B,S)
        batch_size, maximum_elements_num, feature_size = input_data.shape
        clss = input_data[:, :, :-4].unsqueeze(-1)  # (B,S,10,1)
        geo = input_data[:, :, -4:].unsqueeze(-2)  # (B,S,1,4)

        modified = torch.matmul(clss, geo).permute(
            0, 3, 1, 2).contiguous()  # (B,4,S,10)

        # Encoder
        h0_0 = self.encoder_batch_norm0(
            self.encoder_fc0(modified))  # (B,dim*4,S,10)
        h0_1 = F.relu(self.encoder_batch_norm1(
            self.encoder_fc1(modified)))  # (B,dim,S,10)
        h0_2 = F.relu(self.encoder_batch_norm2(
            self.encoder_fc2(h0_1)))  # (B,dim,S,10)
        h0_3 = self.encoder_batch_norm3(
            self.encoder_fc3(h0_2))  # (B,dim*4,S,10)
        encoded = F.relu(h0_0+h0_3)  # (B,dim*4,S,10)
        encoded = encoded.permute(0, 2, 3, 1).contiguous()  # (B,S,10,dim*4)
        encoded = encoded.reshape(
            batch_size, self.element_num, -1, 1)  # (B,S,dim*4*10,1)
        encoded = encoded.permute(0, 2, 1, 3).contiguous()  # (B,dim*4*10,S,1)

        # Self relation
        """transformed = self.transformer_encoder(
            src=encoded.permute(1, 0, 2), src_key_padding_mask=input_length)  # (S,B,dim*4*10)
        transformed = transformed.permute(1, 2, 0).contiguous(
        ).unsqueeze(-1)  # (B,dim*4*10,S,1)"""
        # Stacked relation module
        relation0 = F.relu(self.relation_bn0(self.relation_nonLocal0(encoded)))
        relation1 = F.relu(self.relation_bn1(
            self.relation_nonLocal1(relation0)))
        residual_block1 = encoded + relation1
        relation2 = F.relu(self.relation_bn2(
            self.relation_nonLocal2(residual_block1)))
        relation3 = F.relu(self.relation_bn3(
            self.relation_nonLocal3(relation2)))
        transformed = residual_block1 + relation3

        # Decoder
        h1_0 = self.decoder_batch_norm0(
            self.decoder_fc0(transformed))  # (B,dim*4,S,1)
        h1_1 = F.relu(self.decoder_batch_norm1(
            self.decoder_fc1(transformed)))  # (B,dim,S,1)
        h1_2 = F.relu(self.decoder_batch_norm2(
            self.decoder_fc2(h1_1)))  # (B,dim,S,1)
        h1_3 = self.decoder_batch_norm3(
            self.decoder_fc3(h1_2))  # (B,dim*4,S,1)
        decoded = F.relu(h1_0+h1_3)  # (B,dim*4,S,1)

        # Branch
        # syn_cls = self.branch_fc0(decoded)+ self.cls.permute(0,2,1,3)#大跨residual connect#(16,10,8,1)
        #syn_cls = F.relu(syn_cls)
        syn_cls = clss  # ((B,S,10,1)
        syn_turb = self.branch_fc1(decoded)  # (B,4,S,1)
        syn_turb[:, 2, :, :] = 0  # 保持面积不变
        syn_geo = syn_turb.permute(0, 2, 1, 3).contiguous()\
            + geo.permute(0, 1, 3,
                          2).contiguous()  # 大跨residual connect#(B,S,4,1)
        #syn_geo = (syn_geo*element_std)+element_mean

        # Synthesized layout
        res = torch.cat((syn_cls, syn_geo), 2).squeeze(-1)  # (B,S,14)

        # Remove redundancy
        '''mask_l = [torch.cat([torch.ones(l,feature_size),torch.zeros(maximum_elements_num-l,feature_size)]) for l in input_length.cpu().numpy()]
        mask = torch.stack(mask_l)
        mask = mask.to(self.device)'''
        #res = res.masked_fill(input_length.unsqueeze(-1).repeat(1, 1, res.shape[-1]), value=torch.tensor(0))

        return (res, input_length)


class WireframeDiscriminator(nn.Module):
    def __init__(self, dataset, renderer, cfg=None):
        super(WireframeDiscriminator, self).__init__()
        self.class_num = dataset.enc_len
        self.element_num = dataset.maximum_elements_num
        self.render_size = renderer.render_size
        self.renderer = renderer
        num_resblocks = 3 if cfg is None else cfg.MODEL.DISCRIMINATOR.NUM_RESBLOCKS
        in_channels = self.class_num
        out_channels = 64

        self.cnn = nn.ModuleList()
        for _ in range(num_resblocks):
            #resblock = ResidualBlock(in_channels, out_channels)
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5,
                          stride=2, padding=2, bias=False),  # padding=2,same
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
            self.cnn.append(layer)
            in_channels = out_channels
            out_channels *= 2

        self.classifier = nn.Sequential(
            nn.Conv2d(
                256,
                1024,
                kernel_size=self.render_size >> num_resblocks,
                stride=1,
                bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        for param in self.named_parameters():
            torch.nn.init.normal_(param[1].data, mean=0.0, std=0.02)

    def forward(self, input_data, input_length):
        #input_data = input[0]
        batch_size = input_data.shape[0]
        rendered = self.renderer.render(input_data)
        x = rendered
        for m in self.cnn:
            x = m(x)
        #reshaped = conved.reshape(batch_size,-1)
        output = self.classifier(x)
        return output.squeeze()

# Residual block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size >> 1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size >> 1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size >> 1,
            bias=False
        )
        self.bn_downsample = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.conv_downsample(x)
        residual = self.bn_downsample(residual)
        out = out + residual  # inplace add
        out = self.lrelu(out)
        return out


def weight_init(m):
    # weight_initialization: important for wgan
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Linear') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)


class renderer_g2v():
    def __init__(self, render_size, class_num):
        super(renderer_g2v, self).__init__()
        self.render_size = render_size
        self.class_num = class_num

    def render(self, elements):
        try:
            elements = torch.from_numpy(elements)
        except:
            pass
        device = elements.device
        batch_size = elements.shape[0]
        num_elements = elements.shape[1]  # batch_size,46,13
        rects = elements[:, :, -4:]  # batch_size,46,4
        class_ = elements[:, :, :-4]  # batch_size,46,9

        coor_x = torch.arange(self.render_size).unsqueeze(1).expand(self.render_size, self.render_size).T\
            .expand(batch_size, num_elements, self.render_size, self.render_size).to(device)
        coor_y = coor_x.transpose(2, 3).contiguous().to(device)

        x = rects[:, :, 0].reshape(
            batch_size, -1, 1, 1)*self.render_size  # batch_size,46,1,1
        y = rects[:, :, 1].reshape(batch_size, -1, 1, 1)*self.render_size
        area_root = rects[:, :, 2].reshape(
            batch_size, -1, 1, 1)*self.render_size
        w = rects[:, :, 3].reshape(batch_size, -1, 1, 1)*self.render_size

        h = area_root**2/w
        try:
            h[h != h] = 0
        except:
            pass

        x0 = x-0.5*w
        x1 = x+0.5*w
        y0 = y-0.5*h
        y1 = y+0.5*h
        '''x0 = rects[:,:,0].reshape(batch_size,-1,1,1)*self.rendered_size #batch_size,46,1,1
        y0 = rects[:,:,1].reshape(batch_size,-1,1,1)*self.rendered_size
        x1 = rects[:,:,2].reshape(batch_size,-1,1,1)*self.rendered_size
        y1 = rects[:,:,3].reshape(batch_size,-1,1,1)*self.rendered_size'''

        def k(c, c_r):
            temp = torch.abs(c-c_r)
            temp = torch.stack((1-temp, torch.zeros_like(c)))
            return torch.max(temp, 0).values

        def b0(c, c_r):
            temp = c-c_r
            temp = torch.stack((temp, torch.zeros_like(c)))
            temp = torch.max(temp, 0).values
            temp = torch.stack((temp, torch.ones_like(c)))
            return torch.min(temp, 0).values

        def b1(c_r, c):
            temp = c_r-c
            temp = torch.stack((temp, torch.zeros_like(c)))
            temp = torch.max(temp, 0).values
            temp = torch.stack((temp, torch.ones_like(c)))
            return torch.min(temp, 0).values

        # left edge
        l0 = k(coor_x, x0).mul(b0(coor_y, y0)).mul(b1(y1, coor_y))

        # right edge
        l1 = k(coor_x, x1).mul(b0(coor_y, y0)).mul(b1(y1, coor_y))

        # top edge
        l2 = k(coor_y, y0).mul(b0(coor_x, x0)).mul(b1(x1, coor_x))

        # bottom edge
        l3 = k(coor_y, y1).mul(b0(coor_x, x0)).mul(b1(x1, coor_x))

        rendered = torch.max(torch.stack((l0, l1, l2, l3)), 0).values
        '''F_ = torch.sum(\
                    class_.reshape(batch_size,num_elements,1,1,class_num)\
                    .mul(rendered.reshape(batch_size,num_elements,rendered_size,rendered_size,1))\
                    ,dim=1).permute(0,3,1,2)'''
        F_ = torch.max(
            class_.reshape(batch_size, num_elements, 1, 1, self.class_num)
            .mul(rendered.reshape(batch_size, num_elements, self.render_size, self.render_size, 1)), dim=1)[0].permute(0, 3, 1, 2).contiguous()
        return F_
