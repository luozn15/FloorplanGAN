import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

# Stacked relation module


class Generator(nn.Module):
    def __init__(self, dataset):  # feature_size, class_num, element_num):
        super(Generator, self).__init__()

        self.class_num = dataset.enc_len  # 10
        self.element_num = dataset.maximum_elements_num  # 8
        dim = 128
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim*4*self.class_num, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3)
        '''self.transformer = nn.Transformer(d_model=2560,
                                            nhead=10,
                                            num_encoder_layers=3,
                                            num_decoder_layers =3)'''

        # Decoder, two fully connected layers.
        self.decoder_fc0 = nn.Conv2d(
            dim*4, dim*4, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm0 = nn.BatchNorm2d(dim*4)
        self.decoder_fc1 = nn.Conv2d(
            dim*4, dim, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm1 = nn.BatchNorm2d(dim)
        self.decoder_fc2 = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm2 = nn.BatchNorm2d(dim)
        self.decoder_fc3 = nn.Conv2d(
            dim, dim*4, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm3 = nn.BatchNorm2d(dim*4)

        self.decoder_fc4 = nn.Conv2d(
            dim*4, 4, kernel_size=1, stride=1, bias=False)
        self.decoder_batch_norm4 = nn.BatchNorm2d(4)

        # Branch of class
        # self.branch_fc0 = nn.Conv2d(1024, self.class_num,kernel_size=1,stride=1) #(16,10,8,1)
        #self.sigmoid_brach0 = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)
        # Branch of position
        # self.branch_fc1 = nn.Conv2d(dim*4, 4 ,kernel_size=1,stride=1) #(16,4,8,1)
        self.branch_fc1 = nn.Linear(self.class_num, 1)  # (16,4,8,1)
        self.sigmoid_brach1 = nn.Sigmoid()

    def forward(self, input_data, input_length):  # (B,S,14),(B,S)
        batch_size, maximum_elements_num, feature_size = input_data.shape
        clss = input_data[:, :, :-4].unsqueeze(-1)  # (B,S,10,1)
        geo = input_data[:, :, -4:].unsqueeze(-2)  # (B,S,1,4)

        modified = torch.matmul(clss, geo).permute(0, 3, 1, 2)  # (B,4,S,10)

        # Encoder
        h0_0 = self.encoder_batch_norm0(
            self.encoder_fc0(modified))  # (B,dim*4,S,10)
        h0_1 = F.relu(self.encoder_batch_norm1(
            self.encoder_fc1(modified)))  # (B,dim,S,10)
        h0_2 = F.relu(self.encoder_batch_norm2(
            self.encoder_fc2(h0_1)))  # (B,dim,S,10)
        h0_3 = self.encoder_batch_norm3(
            self.encoder_fc3(h0_2))  # (B,dim*4,S,10)
        #out = F.relu(self.encoder_fc1(normed_data))
        #out = F.relu(self.encoder_fc2(out))
        encoded = F.relu(h0_0+h0_3)  # (B,dim*4,S,10)
        encoded = encoded.permute(2, 0, 1, 3)  # (S,B,dim*4,10)
        encoded = encoded.reshape(
            self.element_num, batch_size, -1)  # (S,B,dim*4*10)

        # Stacked relation module
        '''relation0 = F.relu(self.relation_bn0(self.relation_nonLocal0(encoded)))
        relation1 = F.relu(self.relation_bn1(self.relation_nonLocal1(relation0)))
        residual_block1 = encoded + relation1
        relation2 = F.relu(self.relation_bn2(self.relation_nonLocal2(residual_block1)))
        relation3 = F.relu(self.relation_bn3(self.relation_nonLocal3(relation2)))
        residual_block2 = residual_block1 + relation3'''

        transformed = self.transformer_encoder(
            src=encoded, src_key_padding_mask=input_length)  # (S,B,dim*4*10)
        transformed = transformed.reshape(
            transformed.shape[0], transformed.shape[1], -1, self.class_num)  # (S,B,dim*4,10)
        transformed = transformed.permute(1, 2, 0, 3)  # (B,dim*4,S,10)
        #self.relation = F.relu(self.relation_bn1(self.relation+self.relation_nonLocal1(self.relation)))

        # Decoder
        h1_0 = self.decoder_batch_norm0(
            self.decoder_fc0(transformed))  # (B,dim*4,S,10)
        h1_1 = F.relu(self.decoder_batch_norm1(
            self.decoder_fc1(transformed)))  # (B,dim,S,10)
        h1_2 = F.relu(self.decoder_batch_norm2(
            self.decoder_fc2(h1_1)))  # (B,dim,S,10)
        h1_3 = self.decoder_batch_norm3(
            self.decoder_fc3(h1_2))  # (B,dim*4,S,10)
        decoded = F.relu(h1_0+h1_3)  # (B,dim*4,S,10)

        decoded = F.relu(self.decoder_batch_norm4(
            self.decoder_fc4(decoded)))  # (B,4,S,10)

        # Branch
        # syn_cls = self.branch_fc0(decoded)+ self.cls.permute(0,2,1,3)#大跨residual connect#(16,10,8,1)
        #syn_cls = F.relu(syn_cls)
        syn_cls = clss.permute(0, 2, 1, 3)  # ((B,10,S,1)

        syn_geo = self.sigmoid_brach1(self.branch_fc1(
            decoded)) - 0.5 + geo.permute(0, 3, 1, 2)  # 大跨residual connect#(B,4,S,1)
        #syn_geo = (syn_geo*element_std)+element_mean

        # Synthesized layout
        res = torch.cat((syn_cls, syn_geo), 1).squeeze().permute(
            0, 2, 1)  # (B,S,14)

        # Remove redundancy
        '''mask_l = [torch.cat([torch.ones(l,feature_size),torch.zeros(maximum_elements_num-l,feature_size)]) for l in input_length.cpu().numpy()]
        mask = torch.stack(mask_l)
        mask = mask.to(self.device)'''
        res = res.masked_fill(
            input_length.unsqueeze(-1).repeat(1, 1, res.shape[-1]), value=torch.tensor(0))

        return (res, input_length)


class Generator2(nn.Module):
    def __init__(self, dataset):  # feature_size, class_num, element_num):
        super(Generator2, self).__init__()

        self.class_num = dataset.enc_len  # 10
        self.element_num = dataset.maximum_elements_num  # 8
        dim = 128
        # embedder for ctg: two fully connected layers, input layout Z.
        self.ctg_fc0 = nn.Linear(10, dim*4, bias=False)
        self.ctg_norm0 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc1 = nn.Linear(10, dim, bias=False)
        self.ctg_norm1 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc2 = nn.Linear(dim, dim, bias=False)
        self.ctg_norm2 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.ctg_norm3 = nn.BatchNorm1d(self.element_num)

        # embedder for geo: two fully connected layers, input layout Z.
        self.geo_fc0 = nn.Linear(4, dim*4, bias=False)  # 4 是 (xc,w,yc,h) 的位数
        self.geo_norm0 = nn.BatchNorm1d(self.element_num)
        self.geo_fc1 = nn.Linear(4, dim, bias=False)
        self.geo_norm1 = nn.BatchNorm1d(self.element_num)
        self.geo_fc2 = nn.Linear(dim, dim, bias=False)
        self.geo_norm2 = nn.BatchNorm1d(self.element_num)
        self.geo_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.geo_norm3 = nn.BatchNorm1d(self.element_num)

        # encoder
        self.encoder_fc0 = nn.Linear(dim*4*2, dim*4, bias=False)
        self.encoder_norm0 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc1 = nn.Linear(dim*4*2, dim, bias=False)
        self.encoder_norm1 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc2 = nn.Linear(dim, dim, bias=False)
        self.encoder_norm2 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.encoder_norm3 = nn.BatchNorm1d(self.element_num)

        # Batchsize=16, Chanel=dim*4 ,S=8, O=10
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim*4, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3)

        # Decoder, two fully connected layers.
        self.decoder_fc0 = nn.Linear(dim*4, dim*4, bias=False)
        self.decoder_norm0 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc1 = nn.Linear(dim*4, dim, bias=False)
        self.decoder_norm1 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc2 = nn.Linear(dim, dim, bias=False)
        self.decoder_norm2 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.decoder_norm3 = nn.BatchNorm1d(self.element_num)

        # Branch of category
        self.branch_ctg = nn.Linear(dim*4, self.class_num, bias=True)  # B,S,10
        self.sigmoid_ctg = nn.Sigmoid()

        # Branch of geo
        self.branch_geo = nn.Linear(dim*4, 4, bias=True)  # B,S,4
        self.tanh_geo = nn.Tanh()

    def forward(self, input_data, input_length):  # (B,S,14),(B,S)
        batch_size, maximum_elements_num, feature_size = input_data.shape
        ctg = input_data[:, :, :-4]  # B,S,10
        geo = input_data[:, :, -4:]  # B,S,4

        # embed_ctg
        embed_ctg_0 = self.ctg_norm0(self.ctg_fc0(ctg))  # B,S,dim*4
        embed_ctg_1 = F.relu(self.ctg_norm1(self.ctg_fc1(ctg)))  # B,S,dim
        embed_ctg_2 = F.relu(self.ctg_norm2(
            self.ctg_fc2(embed_ctg_1)))  # B,S,dim
        embed_ctg_3 = self.ctg_norm3(self.ctg_fc3(embed_ctg_2))  # B,S,dim*4
        embed_ctg = F.relu(embed_ctg_0+embed_ctg_3)  # B,S,dim*4

        # embed_geo
        embed_geo_0 = self.geo_norm0(self.geo_fc0(geo))  # B,S,dim*4
        embed_geo_1 = F.relu(self.geo_norm1(self.geo_fc1(geo)))  # B,S,dim
        embed_geo_2 = F.relu(self.geo_norm2(
            self.geo_fc2(embed_geo_1)))  # B,S,dim
        embed_geo_3 = self.geo_norm3(self.geo_fc3(embed_geo_2))  # B,S,dim*4
        embed_geo = F.relu(embed_geo_0+embed_geo_3)  # B,S,dim*4

        concat = torch.cat([embed_ctg, embed_geo], -1)  # B,S,dim*4*2

        # encode
        encode_0 = self.encoder_norm0(self.encoder_fc0(concat))  # B,S,dim*4
        encode_1 = F.relu(self.encoder_norm1(
            self.encoder_fc1(concat)))  # B,S,dim
        encode_2 = F.relu(self.encoder_norm2(
            self.encoder_fc2(encode_1)))  # B,S,dim
        encode_3 = self.encoder_norm3(self.encoder_fc3(encode_2))  # B,S,dim*4
        encoded = F.relu(encode_0+encode_3)  # B,S,dim*4
        attention_input = encoded.permute(1, 0, 2)  # (S,B,dim*4)

        # attention
        attention_output = self.transformer_encoder(
            src=attention_input, src_key_padding_mask=input_length)  # (S,B,dim*4)
        attention_output = attention_output.permute(1, 0, 2)  # B,S,dim*4

        # decoder
        decode_0 = self.decoder_norm0(
            self.decoder_fc0(attention_output))  # B,S,dim*4
        decode_1 = F.relu(self.decoder_norm1(
            self.decoder_fc1(attention_output)))  # B,S,dim
        decode_2 = F.relu(self.decoder_norm2(
            self.decoder_fc2(decode_1)))  # B,S,dim
        decode_3 = self.decoder_norm3(self.decoder_fc3(decode_2))  # B,S,dim*4
        decoded = F.relu(decode_0+decode_3)  # B,S,dim*4

        # Branch
        syn_ctg = self.sigmoid_ctg(self.branch_ctg(decoded))  # B,S,10
        # 大跨residual connect#(B,S,4)
        syn_geo = self.tanh_geo(self.branch_geo(decoded)) + geo

        # Synthesized layout
        syn = torch.cat((syn_ctg, syn_geo), -1)  # (B,S,14)

        # Remove redundancy
        res = syn.masked_fill(
            input_length.unsqueeze(-1).repeat(1, 1, syn.shape[-1]), value=torch.tensor(0))

        return (res, input_length)


class VectorDiscriminator(nn.Module):
    def __init__(self, dataset):  # feature_size, class_num, element_num):
        super(VectorDiscriminator, self).__init__()

        self.class_num = dataset.enc_len  # 10
        self.element_num = dataset.maximum_elements_num  # 8
        dim = 128
        # embedder for ctg: two fully connected layers, input layout Z.
        self.ctg_fc0 = nn.Linear(10, dim*4, bias=False)
        self.ctg_norm0 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc1 = nn.Linear(10, dim, bias=False)
        self.ctg_norm1 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc2 = nn.Linear(dim, dim, bias=False)
        self.ctg_norm2 = nn.BatchNorm1d(self.element_num)
        self.ctg_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.ctg_norm3 = nn.BatchNorm1d(self.element_num)

        # embedder for geo: two fully connected layers, input layout Z.
        self.geo_fc0 = nn.Linear(4, dim*4, bias=False)  # 4 是 (xc,w,yc,h) 的位数
        self.geo_norm0 = nn.BatchNorm1d(self.element_num)
        self.geo_fc1 = nn.Linear(4, dim, bias=False)
        self.geo_norm1 = nn.BatchNorm1d(self.element_num)
        self.geo_fc2 = nn.Linear(dim, dim, bias=False)
        self.geo_norm2 = nn.BatchNorm1d(self.element_num)
        self.geo_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.geo_norm3 = nn.BatchNorm1d(self.element_num)

        # encoder
        self.encoder_fc0 = nn.Linear(dim*4*2, dim*4, bias=False)
        self.encoder_norm0 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc1 = nn.Linear(dim*4*2, dim, bias=False)
        self.encoder_norm1 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc2 = nn.Linear(dim, dim, bias=False)
        self.encoder_norm2 = nn.BatchNorm1d(self.element_num)
        self.encoder_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.encoder_norm3 = nn.BatchNorm1d(self.element_num)

        # Batchsize=16, Chanel=dim*4 ,S=8, O=10
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim*4, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3)

        # Decoder, two fully connected layers.
        self.decoder_fc0 = nn.Linear(dim*4, dim*4, bias=False)
        self.decoder_norm0 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc1 = nn.Linear(dim*4, dim, bias=False)
        self.decoder_norm1 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc2 = nn.Linear(dim, dim, bias=False)
        self.decoder_norm2 = nn.BatchNorm1d(self.element_num)
        self.decoder_fc3 = nn.Linear(dim, dim*4, bias=False)
        self.decoder_norm3 = nn.BatchNorm1d(self.element_num)

        # judge
        self.judge_fc0 = nn.Linear(dim*4, 1, bias=True)  # B,S,1
        self.judge_norm0 = nn.BatchNorm1d(self.element_num)
        self.judge_fc1 = nn.Linear(self.element_num, 1, bias=True)  # B,1
        #self.sigmoid_judge = nn.Sigmoid()

    def forward(self, input_data, input_length):  # (B,S,14),(B,S)
        batch_size, maximum_elements_num, feature_size = input_data.shape
        ctg = input_data[:, :, :-4]  # B,S,10
        geo = input_data[:, :, -4:]  # B,S,4

        # embed_ctg
        embed_ctg_0 = self.ctg_norm0(self.ctg_fc0(ctg))  # B,S,dim*4
        embed_ctg_1 = F.relu(self.ctg_norm1(self.ctg_fc1(ctg)))  # B,S,dim
        embed_ctg_2 = F.relu(self.ctg_norm2(
            self.ctg_fc2(embed_ctg_1)))  # B,S,dim
        embed_ctg_3 = self.ctg_norm3(self.ctg_fc3(embed_ctg_2))  # B,S,dim*4
        embed_ctg = F.relu(embed_ctg_0+embed_ctg_3)  # B,S,dim*4

        # embed_geo
        embed_geo_0 = self.geo_norm0(self.geo_fc0(geo))  # B,S,dim*4
        embed_geo_1 = F.relu(self.geo_norm1(self.geo_fc1(geo)))  # B,S,dim
        embed_geo_2 = F.relu(self.geo_norm2(
            self.geo_fc2(embed_geo_1)))  # B,S,dim
        embed_geo_3 = self.geo_norm3(self.geo_fc3(embed_geo_2))  # B,S,dim*4
        embed_geo = F.relu(embed_geo_0+embed_geo_3)  # B,S,dim*4

        concat = torch.cat([embed_ctg, embed_geo], -1)  # B,S,dim*4*2

        # encode
        encode_0 = self.encoder_norm0(self.encoder_fc0(concat))  # B,S,dim*4
        encode_1 = F.relu(self.encoder_norm1(
            self.encoder_fc1(concat)))  # B,S,dim
        encode_2 = F.relu(self.encoder_norm2(
            self.encoder_fc2(encode_1)))  # B,S,dim
        encode_3 = self.encoder_norm3(self.encoder_fc3(encode_2))  # B,S,dim*4
        encoded = F.relu(encode_0+encode_3)  # B,S,dim*4
        attention_input = encoded.permute(1, 0, 2)  # (S,B,dim*4)

        # attention
        attention_output = self.transformer_encoder(
            src=attention_input, src_key_padding_mask=input_length)  # (S,B,dim*4)
        attention_output = attention_output.permute(1, 0, 2)  # B,S,dim*4

        # decoder
        decode_0 = self.decoder_norm0(
            self.decoder_fc0(attention_output))  # B,S,dim*4
        decode_1 = F.relu(self.decoder_norm1(
            self.decoder_fc1(attention_output)))  # B,S,dim
        decode_2 = F.relu(self.decoder_norm2(
            self.decoder_fc2(decode_1)))  # B,S,dim
        decode_3 = self.decoder_norm3(self.decoder_fc3(decode_2))  # B,S,dim*4
        decoded = F.relu(decode_0+decode_3)  # B,S,dim*4

        # judge
        judge_0 = self.judge_norm0(self.judge_fc0(decoded))  # B,S,1
        judge_1 = self.judge_fc1(judge_0.squeeze())  # B,1

        return judge_1.squeeze()  # B


class WireframeDiscriminator(nn.Module):
    def __init__(self, dataset, rendered_size=64):
        super(WireframeDiscriminator, self).__init__()
        self.class_num = dataset.enc_len
        self.element_num = dataset.maximum_elements_num
        self.rendered_size = rendered_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self.class_num, 64, kernel_size=5, stride=2,
                      padding=2, bias=False),  # padding=2,same
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),  # 32, rendered_size/2, rendered_size/2

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),  # 64, rendered_size/4, rendered_size/4

            nn.Conv2d(128, 256, kernel_size=5,
                      stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            # 128, rendered_size/8, rendered_size/8
            nn.LeakyReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=int(
                self.rendered_size/8), stride=1, bias=False),
            # nn.Linear(in_features=64*16*16,out_features=512,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=True),
            # nn.Linear(in_features=512,out_features=1),
            # nn.Sigmoid() Wasserstein GAN去掉最后一层sigmoid
        )

    def forward(self, input_data, input_length):
        #input_data = input[0]
        batch_size = input_data.shape[0]
        rendered = self.render(input_data)  # batch_size,9,64,64
        conved = self.cnn(rendered)  # batch_size,64,16,16
        #reshaped = conved.reshape(batch_size,-1)
        return self.classifier(conved).squeeze()

    def render(self, elements):
        device = elements.device
        batch_size = elements.shape[0]
        num_elements = elements.shape[1]  # batch_size,46,13
        rects = elements[:, :, -4:]  # batch_size,46,4
        class_ = elements[:, :, :-4]  # batch_size,46,9

        coor_x = torch.arange(self.rendered_size).unsqueeze(1).expand(self.rendered_size, self.rendered_size).T\
            .expand(batch_size, num_elements, self.rendered_size, self.rendered_size).to(device)
        coor_y = coor_x.transpose(2, 3).to(device)

        x = rects[:, :, 0].reshape(
            batch_size, -1, 1, 1)*self.rendered_size  # batch_size,46,1,1
        y = rects[:, :, 1].reshape(batch_size, -1, 1, 1)*self.rendered_size
        w = rects[:, :, 2].reshape(batch_size, -1, 1, 1)*self.rendered_size
        h = rects[:, :, 3].reshape(batch_size, -1, 1, 1)*self.rendered_size

        '''h = area_root**2/w
        try:
            h[h != h] = 0
        except:
            pass'''

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
        F_ = torch.sum(
            class_.reshape(batch_size, num_elements, 1, 1, self.class_num)
            .mul(rendered.reshape(batch_size, num_elements, self.rendered_size, self.rendered_size, 1)), dim=1).permute(0, 3, 1, 2)
        return F_


def weight_init(m):
    # weight_initialization: important for wgan
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Linear') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)


def render(elements, rendered_size, class_num):
    try:
        elements = torch.from_numpy(elements)
    except:
        pass
    device = elements.device
    batch_size = elements.shape[0]
    num_elements = elements.shape[1]  # batch_size,46,13
    rects = elements[:, :, -4:]  # batch_size,46,4
    class_ = elements[:, :, :-4]  # batch_size,46,9

    coor_x = torch.arange(rendered_size).unsqueeze(1).expand(rendered_size, rendered_size).T\
        .expand(batch_size, num_elements, rendered_size, rendered_size).to(device)
    coor_y = coor_x.transpose(2, 3).to(device)

    x = rects[:, :, 0].reshape(batch_size, -1, 1, 1) * \
        rendered_size  # batch_size,46,1,1
    y = rects[:, :, 1].reshape(batch_size, -1, 1, 1)*rendered_size
    w = rects[:, :, 2].reshape(batch_size, -1, 1, 1)*rendered_size
    h = rects[:, :, 3].reshape(batch_size, -1, 1, 1)*rendered_size

    '''h = area_root**2/w
    try:
        h[h != h] = 0
    except:
        pass'''

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
        class_.reshape(batch_size, num_elements, 1, 1, class_num)
        .mul(rendered.reshape(batch_size, num_elements, rendered_size, rendered_size, 1)), dim=1)[0].permute(0, 3, 1, 2)
    return F_


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
        coor_y = coor_x.transpose(2, 3).to(device)

        x = rects[:, :, 0].reshape(
            batch_size, -1, 1, 1)*self.render_size  # batch_size,46,1,1
        y = rects[:, :, 1].reshape(batch_size, -1, 1, 1)*self.render_size
        w = rects[:, :, 2].reshape(batch_size, -1, 1, 1)*self.render_size
        h = rects[:, :, 3].reshape(batch_size, -1, 1, 1)*self.render_size

        '''h = area_root**2/w
        try:
            h[h != h] = 0
        except:
            pass'''

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
            .mul(rendered.reshape(batch_size, num_elements, self.render_size, self.render_size, 1)), dim=1)[0].permute(0, 3, 1, 2)
        return F_
