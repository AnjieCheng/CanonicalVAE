from re import S
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import os
from src.canonicalvq.loss import *

class ImplicitFun(nn.Module):
    def __init__(self, z_dim=256, add_dim=3):
        super(ImplicitFun, self).__init__()
        self.z_dim = z_dim
        input_dim = z_dim+add_dim

        self.unfold1 = mlpAdj(nlatent=input_dim)
        self.unfold2 = mlpAdj(nlatent=z_dim+3)

    def forward(self, z, points):

        num_pts = points.size(1)

        if z.ndim == 1:
            # [512] --> [1, N, 512]
            z = z.unsqueeze(0).unsqueeze(1).repeat(1, num_pts, 1)
        elif z.ndim == 2 and z.size(1) != 0:
            # [32, 512] --> [32, N, 512]
            z = z.unsqueeze(1).repeat(1, num_pts, 1)
        else:
            z = z
        
        pointz = torch.cat((points, z), dim=2).float()

        x1 = self.unfold1(pointz)
        x2 = torch.cat((x1, z), dim=2)
        x3 = self.unfold2(x2)

        return x3


class Mapping2Dto3D(nn.Module):
    def __init__(self, z_dim=256, add_dim=3):
        self.z_dim = z_dim
        self.bottleneck_size = z_dim
        self.input_size = 3
        self.dim_output = 3
        self.hidden_neurons = 512
        self.num_layers = 2
        super(Mapping2Dto3D, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = F.relu

        self.th = nn.Tanh()

    def forward(self, latent, x):
        x = x.transpose(2,1).contiguous()
        x = self.conv1(x) + latent.transpose(2,1).contiguous()
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.th(self.last_conv(x).transpose(2,1).contiguous())


class mlpAdj(nn.Module):
    def __init__(self, nlatent = 1024):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent//2, self.nlatent//4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent//4, 3, 1)

        # self.th = nn.Tanh()
        self.th = nn.Identity()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent) # nn.Identity() 
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2) # nn.Identity() 
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4) # nn.Identity() 

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2,1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x.transpose(2,1)

class FixPointGenerator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.use_bias = True

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=64,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048 * 3,
                      bias=self.use_bias),
        )

    def forward(self, z, _):
        z = z[:,0,:]
        output = self.model(z.squeeze())
        output = output.view(-1, 2048, 3).contiguous()
        return output


###############################################################################

# encoding=utf-8

import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from src.canonicalvq.models.encoder import DGCNN

cudnn.benchnark=True
from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

neg = 0.01
neg_2 = 0.2
class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        # Conv = EqualConv1d if use_eql else nn.Conv1d
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x

def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k).contiguous()
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)

    if return_idx:
        return ee, idx
    return ee


class Attention(nn.Module):
  def __init__(self, ch, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = nn.Conv2d

    self.theta = nn.Conv1d(self.ch, self.ch // 8, 1, bias=False)
    self.phi = nn.Conv1d(self.ch, self.ch // 8, 1, bias=False)
    self.g = nn.Conv1d(self.ch, self.ch // 2, 1, bias=False)
    self.o = nn.Conv1d(self.ch // 2, self.ch, 1, bias=False)
    # Learnable gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = self.phi(x)
    g = self.g(x)
    # Perform reshapes
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)))
    return self.gamma * o + x

class SPGANGenerator(nn.Module):
    def __init__(self, z_dim, add_dim=3, use_local=True, use_tanh=False, norm_z=False, use_attn=False):
        super(SPGANGenerator, self).__init__()
        self.np = 2048
        self.nk = 20//2
        self.nz = z_dim
        self.z_dim = z_dim
        self.off = False
        self.use_attn = False
        self.use_head = False
        self.use_local = use_local
        self.use_tanh = use_tanh
        self.norm_z = norm_z
        self.use_attn = use_attn

        # Conv = EqualConv1d if self.opts.eql else nn.Conv1d
        Conv = nn.Conv1d
        # Linear = EqualLinear if self.opts.eql else nn.Linear
        Linear = nn.Linear

        dim = 128
        self.head = nn.Sequential(
            Conv(add_dim + self.nz, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_attn:
            self.attn = Attention(dim + 512)

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_tanh:
            self.tail = nn.Sequential(
                Conv1d(512+dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(64, 3, 1),
                nn.Tanh()
            )
        else:
            self.tail = nn.Sequential(
                Conv1d(512+dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(64, 3, 1),
                # nn.Tanh()
            )

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(add_dim, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            self.adain1 = AdaptivePointNorm(dim, dim)
            self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            self.EdgeConv1 = EdgeBlock(add_dim, 64, self.nk)
            self.adain1 = AdaptivePointNorm(64, dim)
            self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)


    def forward(self, z, x):
        B,N,_ = x.size()
        if self.norm_z:
            z = z / (z.norm(p=2, dim=-1, keepdim=True)+1e-8)
        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1).contiguous()
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1).contiguous()
        feat_global = feat_global.repeat(1, 1, N)

        if self.use_local:
            feat_cat = torch.cat((feat_global, x2), dim=1)
        else:
            feat_cat = feat_global

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_p = self.tail(feat_cat)                   # Bx3x256
        return x1_p.transpose(1,2).contiguous()

    def interpolate(self, x, z1, z2, selection, alpha, use_latent = False):

        if not use_latent:

            ## interpolation
            z = z1
            z[:, selection == 1] = z1[:, selection == 1] * (1 - alpha) + z2[:, selection == 1] * (alpha)

            B, N, _ = x.size()
            if self.opts.z_norm:
                z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style = torch.cat([x, z], dim=-1)
            style = style.transpose(2, 1).contiguous()
            style = self.head(style)  # B,C,N

        else:
            # interplolation
            B, N, _ = x.size()
            if self.opts.z_norm:
                z1 = z1 / (z1.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                z2 = z2 / (z2.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style_1 = torch.cat([x, z1], dim=-1)
            style_1 = style_1.transpose(2, 1).contiguous()
            style_1 = self.head(style_1)  # B,C,N

            style_2 = torch.cat([x, z2], dim=-1)
            style_2 = style_2.transpose(2, 1).contiguous()
            style_2 = self.head(style_2)  # B,C,N

            style = style_1
            style[:, :, selection == 1] = style_1[:, :, selection == 1] * (1 - alpha) + style_2[:, :, selection == 1] * alpha

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1).contiguous()
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1).contiguous()
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_o = self.tail(feat_cat)  # Bx3x256

        x1_p = pc + x1_o if self.off else x1_o

        return x1_p\

class PrimitiveGrouping(nn.Module):
    def __init__(self, num_groups=64, in_channels=3, learnable=True, regular_points=None, abl=None):
        super(PrimitiveGrouping, self).__init__()
        self.stpts_prob_map = None
        self.num_groups = num_groups
        self.learnable = learnable
        self.abl = abl
        if learnable:
            conv1d_stpts_prob_modules = []
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(128))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128, out_channels=self.num_groups, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_groups))
            self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)
            self.fibonacci_grid_groups_unique = None
            self.regular_points = None
        else:
            raise NotImplementedError

    def sphere_fibonacci_grid_points(self, ng):
        phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0
        theta = np.zeros (ng)
        sphi = np.zeros (ng)
        cphi = np.zeros (ng)
        for i in range ( 0, ng ):
            i2 = 2 * i - (ng - 1) 
            theta[i] = 2.0 * np.pi * float ( i2 ) / phi
            sphi[i] = float ( i2 ) / float ( ng )
            cphi[i] = np.sqrt ( float ( ng + i2 ) * float ( ng - i2 ) ) / float ( ng )
        xg = np.zeros ( ( ng, 3 ) )
        for i in range (0, ng) :
            xg[i,0] = cphi[i] * np.sin ( theta[i] )
            xg[i,1] = cphi[i] * np.cos ( theta[i] )
            xg[i,2] = sphi[i]
        return xg/2

    @torch.no_grad()
    def propagate_feat(self, sphere, weighted_features):
        if self.fibonacci_grid_groups_unique == None:
            self.get_fibonacci_grid_groups_unique(weighted_features)
        
        feat_dim = weighted_features.size(2)

        stpts_prob_map = self.conv1d_stpts_prob(sphere.permute(0, 2, 1).contiguous())
        batch_size, origin_num_of_groups, n_points = stpts_prob_map.size()
        blank_weighted_features = torch.zeros([batch_size, origin_num_of_groups, feat_dim]).to(weighted_features)

        idc = self.fibonacci_grid_groups_unique[None, :, None].expand(batch_size, -1, feat_dim)
        new_weighted_features = blank_weighted_features.scatter_(1, idc, weighted_features)
        new_weighted_features = new_weighted_features.transpose(1,2).contiguous() 

        scattered = torch.zeros_like(stpts_prob_map).scatter_(1, torch.argmax(stpts_prob_map, dim=1, keepdims=True), 1.)
        scattered_features = torch.sum(scattered[:, None, :, :] * new_weighted_features[:, :, :, None], dim=2).transpose(1,2).contiguous()
        return scattered_features

    @torch.no_grad()
    def get_groups(self, sphere):
        # global_x, local_x = self.encoder(sphere)
        stpts_prob_map = self.conv1d_stpts_prob(sphere.permute(0, 2, 1).contiguous())
        groups = torch.argmax(stpts_prob_map, dim=1)
        return groups

    def get_fibonacci_grid_groups_unique(self, placeholder, query_points=5000):
        fibonacci_grid = torch.tensor(self.sphere_fibonacci_grid_points(query_points)).to(placeholder)
        self.fibonacci_grid = fibonacci_grid
        fibonacci_grid_groups = self.get_groups(fibonacci_grid.unsqueeze(0))
        self.fibonacci_grid_groups = fibonacci_grid_groups
        self.fibonacci_grid_groups_unique = torch.unique(fibonacci_grid_groups[0].cpu(), sorted=False).to(placeholder).long()
        self.num_unique_groups = self.fibonacci_grid_groups_unique.size(0)

        if self.abl == "flip":
            self.fibonacci_grid_groups_unique = torch.flip((self.fibonacci_grid_groups_unique), dims=[0])
        if self.abl == "random":
            self.fibonacci_grid_groups_unique = torch.unique(fibonacci_grid_groups[0].cpu(), sorted=True).to(placeholder).long()
        # print(self.fibonacci_grid_groups_unique.shape)

    @torch.no_grad()
    def reorder_weighted_features(self, weighted_features, shape=None):
        if self.fibonacci_grid_groups_unique == None:
            self.get_fibonacci_grid_groups_unique(weighted_features)
        return weighted_features[:, self.fibonacci_grid_groups_unique.long(),:]


    def forward(self, sphere, shape, features=None, fuse_by="max", reorder=False):
        batch_size, n_points = shape.size(0), shape.size(1)
        stpts_prob_map = self.conv1d_stpts_prob(sphere.permute(0, 2, 1).contiguous())
        # stpts_prob_map = stpts_prob_map.clamp(1e-15, 1-1e-15)
        # -------------------------------- #
        stpts_prob_map_soft = F.softmax(stpts_prob_map, dim=2)
        # -------------------------------- #
        weighted_xyz = torch.sum(stpts_prob_map_soft[:, :, :, None] * sphere[:, None, :, :], dim=2)
        weighted_folded = torch.sum(stpts_prob_map_soft[:, :, :, None] * shape[:, None, :, :], dim=2)
        groups = torch.argmax(stpts_prob_map, dim=1)

        # get one-hot
        scattered = torch.zeros_like(stpts_prob_map).scatter_(1, torch.argmax(stpts_prob_map, dim=1, keepdims=True), 1.)

        features = features.transpose(1,2).contiguous()
        if fuse_by == "sum":
            new_features = torch.sum(stpts_prob_map_soft[:, None, :, :].detach() * features[:, :, None, :], dim=2)
            group_features = torch.sum(scattered[:, None, :, :].detach() * new_features[:, :, None, :], dim=3)
        elif fuse_by == "max":
            group_features = torch.max(scattered[:, None, :, :].detach() * features[:, :, None, :], dim=3)[0]

        # get one-hot features
        scattered_features = torch.sum(scattered[:, None, :, :] * group_features[:, :, :, None], dim=2).transpose(1,2).contiguous()
        self.get_fibonacci_grid_groups_unique(sphere)

        return stpts_prob_map_soft, weighted_xyz, groups, group_features.transpose(1,2).contiguous(), scattered_features, weighted_folded

    def remap_grouped_features(self, sphere, grouped_features):
        grouped_features = grouped_features.permute(0, 2, 1).contiguous()
        stpts_prob_map = self.conv1d_stpts_prob(sphere.permute(0, 2, 1).contiguous())
        scattered = torch.zeros_like(stpts_prob_map).scatter_(1, torch.argmax(stpts_prob_map, dim=1, keepdims=True), 1.)
        scattered_features = torch.sum(scattered[:, None, :, :] * grouped_features[:, :, :, None], dim=2).transpose(1,2).contiguous()
        return scattered_features
