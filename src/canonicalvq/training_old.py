import os
from re import I
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict, defaultdict
import scipy.io
import scipy.spatial
from einops import repeat

import torch
import torch.nn.functional as F

from src.training import BaseTrainer
from src.utils.template import SphereTemplate, SquareTemplate
from src.utils import visualize as vis

from src.utils.common import *
from src.canonicalvq.loss import *
from src.canonicalvq.models.transformer.mingpt import sample_with_past

class Trainer(BaseTrainer):
    ''' Trainer object for the Implicit Network.
    Args:
        model (nn.Module): Implicit Network model
        device (device): pytorch device
        vis_dir (str): visualization directory
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, device=None, vis_dir=None, eval_sample=False, config=None):
        self.model = model
        self.optimizer = self.model.configure_optimizers(config)

        self.device = device
        self.vis_dir = vis_dir
        self.eval_sample = eval_sample
        self.smoothed_total_loss = 0
        self.config = config
        self.template = SphereTemplate(device=device)
        # self.template = SquareTemplate(device=device)
        self.start_visdom()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, epoch_it, it):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, epoch_it, it)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.PG.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data, epoch_it, save=None):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''

        device = self.device
        self.model.eval()

        shape = data.get('set').to(self.device).float()
        shape_loc, shape_scale  = data.get('loc').to(device).float(), data.get('scale').to(device).float()
        # shape_m, shape_s = data.get('mean').to(device).float(), data.get('std').to(device).float()

        batch_size, n_points = shape.size(0), shape.size(1)

        with torch.no_grad():
            g_latent, l_latent = self.model.Encoder(shape)

        grid = self.template.get_regular_points(level=4)
        batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
        g_latent_stacked = g_latent.unsqueeze(1).expand(-1, grid.size(0), -1)

        with torch.no_grad():
            global_rec_shape = self.model.Fold(g_latent_stacked, batch_p_2d)
        with torch.no_grad():
            unfold_pts = self.model.Unfold(l_latent, shape)

        if self.config['training']['mode'] == "stage2":
            if self.config['training']['stage2']['normalize_unfold']:
                f_latent_input = l_latent
                f_pts_input = (unfold_pts / (unfold_pts.norm(p=2, dim=-1, keepdim=True)+1e-8))
            elif self.config['training']['stage2']['chamfer_nn']:
                f_latent_input, shape_order, standard_order = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)
                f_pts_input = batch_p_2d

            # with torch.no_grad():
            #     local_rec_shape_from_unfold = self.model.Fold_f(f_latent_input, f_pts_input.detach()) 

            f_latent_input, _, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)

            with torch.no_grad():
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.model.PG(batch_p_2d, global_rec_shape, f_latent_input)
                # import pdb; pdb.set_trace()
                if self.config['training']['stage1']['vq-enabled']:
                    VQ_input = scattered_features.permute(0,2,1).view(batch_size, -1, 2, n_points//2)
                    loss_vq, scattered_features, perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                    scattered_features= scattered_features.view(batch_size, -1, 2 *(n_points//2)).permute(0,2,1)

            with torch.no_grad():
                local_rec_shape_from_standard = self.model.Fold_f(scattered_features, batch_p_2d.detach()) 

                batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())
                _, _, stpts_groups_, _, scattered_features, stpts_xyz_folded_ = self.model.PG(batch_p_2d_in_shape_order, shape, l_latent)
                

                if self.config['training']['stage1']['vq-enabled']:
                    VQ_input = scattered_features.permute(0,2,1).view(batch_size, -1, 2, n_points//2)
                    loss_vq, scattered_features, perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                    scattered_features= scattered_features.view(batch_size, -1, 2 *(n_points//2)).permute(0,2,1)

                local_rec_shape_from_unfold = self.model.Fold_f(scattered_features, batch_p_2d_in_shape_order) 


        # if self.config['training']['mode'] == "vq":
        #     with torch.no_grad():
        #         g_feature, l_latent = self.model.Encoder(shape)

        #     grid = self.template.get_regular_points(level=4)
        #     # grid = self.template.get_random_points(torch.Size((1, 3, 45*45)))
        #     batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
        #     # with torch.no_grad():
        #     #     self_rec_shape = self.model.Fold(g_feature.unsqueeze(1).expand(-1, grid.size(0), -1), batch_p_2d)

        #     # --------- Fold with Local Features ---------
        #     if self.config['model']['latent'] == "local": 

        #         with torch.no_grad():
        #             unfold_pts = self.model.Unfold(l_latent, shape)

        #         if self.config['model']['normalize_unfold']:
        #             unfold_pts_norm = (unfold_pts / (unfold_pts.norm(p=2, dim=-1, keepdim=True)+1e-8))
        #         else:
        #             unfold_pts_norm = unfold_pts

        #         l_latent_query = l_latent
        #         with torch.no_grad():
        #             self_rec_shape = self.model.Fold_f(l_latent_query, unfold_pts_norm.detach()) 

        #         unfold_idc, _ = match_source_to_target_points(batch_p_2d.detach(), unfold_pts_norm.detach(), device=self.device)
        #         l_latent_query = batched_index_select(l_latent, 1, unfold_idc)

        #         with torch.no_grad():
        #             self_rec_shape = self.model.Fold_f(l_latent_query, batch_p_2d.detach()) 

        # --------- Encode ---------
        # with torch.no_grad():
        #     g_latent = self.model.Encoder(shape)

        # # --------- Reshape to [B,C,H,W] ---------
        # g_latent = g_latent.view(batch_size, self.model.Fold.z_dim, 1, 1)

        # ---------    VQ   ---------
        # _, g_latent_vq, _, _ = self.model.VQ(g_latent) # [B,C,H,W]

        # --------- get UV pts ---------
        # grid = self.template.get_regular_points(npoints=45*45, level=4)
        # batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)

        # grid = self.template.get_random_points(torch.Size((1, 3, 45*45))).squeeze().transpose(0,1)
        # batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)

        # --------- Interpolate Local Features ---------
        # g_latent_interpolated = F.interpolate(g_latent_vq, scale_factor=4.5, mode='bilinear', align_corners=True)
        # g_latent_interpolated = g_latent_interpolated.view(batch_size, 16, 45*45).transpose(2,1)  # [B, HxW, C]

        # g_latent_interpolated = g_latent.view(batch_size, self.model.Fold.z_dim).unsqueeze(1).expand(-1,45*45,-1)
        # batch_p_2d = batch_p_2d.transpose(2,1).contiguous()

        # --------- Fold with Local Features ---------
        # with torch.no_grad():
        #     self_rec_shape = self.model.Fold(g_latent_interpolated, batch_p_2d) 

        if self.config['training']['mode'] == "transformer":
            # --------- [Gen] Transformer Sample Sequence ---------
            c_indices = repeat(torch.tensor([self.model.sos_token]), '1 -> b 1', b=batch_size).to(device)  # sos token
            index_sample = sample_with_past(c_indices, self.model.Transformer, steps=100)
            
            # --------- [Gen] Query VQ Codebook ---------
            bhwc = (batch_size,10,10,16)
            quant_z = self.model.VQ.get_codebook_entry(index_sample.reshape(-1), shape=bhwc)

            # --------- [Gen] Interpolate Local Features ---------
            g_latent_interpolated_g = F.interpolate(quant_z, scale_factor=4.5, mode='bilinear', align_corners=True)
            g_latent_interpolated_g = g_latent_interpolated_g.view(batch_size, 16, 45*45).transpose(2,1)  # [B, HxW, C]

            # --------- [Gen] Fold with Local Features ---------
            gen_pts = self.model.Fold(g_latent_interpolated_g, batch_p_2d)      
        
        # de-normalize
        # gdt_shape = denormalize(shape, shape_m, shape_s)
        # rec_shape = denormalize(self_rec_shape, shape_m, shape_s)
        # if self.config['training']['mode'] == "transformer":
        #     gen_shape = denormalize(gen_pts, shape_m, shape_s)

        # de-standardize
        gdt_shape = denormalize(shape, shape_loc, shape_scale)

        if self.config['training']['mode'] == "stage3":
            gen_shape = denormalize(gen_pts, shape_loc, shape_scale)
            rt = {
                'gdt_shape': gdt_shape,
                'gen_shape': gen_shape,
                # 'rec_shape': rec_shape,
            }
        elif self.config['training']['mode'] == "stage2":
            local_rec_shape_from_standard = denormalize(local_rec_shape_from_standard, shape_loc, shape_scale)
            local_rec_shape_from_unfold = denormalize(local_rec_shape_from_unfold, shape_loc, shape_scale)
            global_rec_shape = denormalize(global_rec_shape, shape_loc, shape_scale)
            rt = {
                'gdt_shape': gdt_shape,
                'rec_shape': local_rec_shape_from_standard,
                'local_rec_shape_from_unfold': local_rec_shape_from_unfold,
                'global_rec_shape': global_rec_shape,
            }
        else:
            global_rec_shape = denormalize(global_rec_shape, shape_loc, shape_scale)
            rt = {
                'gdt_shape': gdt_shape,
                'rec_shape': global_rec_shape,
            }

        return rt        

    def visualize(self, data, logger=None, it=None, epoch_it=None):
        ''' Performs a visualization step for the data.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        self.model.eval()

        shape = data.get('set').to(self.device).float()
        # shape_loc, shape_scale  = data.get('loc').to(device).float(), data.get('scale').to(device).float()
        # shape_m, shape_s = data.get('mean').to(device).float(), data.get('std').to(device).float()
        batch_size, n_points = shape.size(0), shape.size(1)

        with torch.no_grad():
            g_latent, l_latent = self.model.Encoder(shape)

        grid = self.template.get_regular_points(level=4)
        batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
        g_latent_stacked = g_latent.unsqueeze(1).expand(-1, grid.size(0), -1)

        with torch.no_grad():
            global_rec_shape = self.model.Fold(g_latent_stacked, batch_p_2d)

        with torch.no_grad():
            unfold_pts = self.model.Unfold(l_latent, shape)

        cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)

        with torch.no_grad():
            unfold_rec_shape = self.model.Fold(g_latent_stacked, unfold_pts)
            cross_rec_shape = self.model.Fold(g_latent_stacked, cross_unfold_pts)

        if self.config['training']['mode'] == "stage2":
            if self.config['training']['stage2']['normalize_unfold']:
                f_latent_input = l_latent
                f_pts_input = (unfold_pts / (unfold_pts.norm(p=2, dim=-1, keepdim=True)+1e-8))
            elif self.config['training']['stage2']['chamfer_nn']:
                f_latent_input, shape_order, standard_order = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)
                f_pts_input = batch_p_2d

            with torch.no_grad():
                local_rec_shape_from_unfold = self.model.Fold_f(f_latent_input, f_pts_input.detach()) 

            f_latent_input, _, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)

            with torch.no_grad():
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features = self.model.PG(batch_p_2d, f_latent_input)

                # stpts_xyz = (stpts_xyz / (stpts_xyz.norm(p=2, dim=-1, keepdim=True)))
                stpts_fold_input = torch.cat([stpts_xyz, batch_p_2d], dim=1)
                g_latent_stpts = g_latent.view(batch_size, self.model.Fold.z_dim).unsqueeze(1).expand(-1, stpts_fold_input.size(1), -1)

                stpts_xyz_folded = self.model.Fold(g_latent_stpts, stpts_fold_input)[:,:stpts_xyz.size(1),:]

            with torch.no_grad():
                local_rec_shape_from_standard = self.model.Fold_f(scattered_features, f_pts_input) 

        for b in trange(batch_size):
            self.visualizer.show_pointclouds(points=shape[b], title="input_{:02d}".format(b))
            self.visualizer.show_pointclouds(points=global_rec_shape[b], title="global_rec_{:02d}".format(b))

            vis.visualize_pointcloud(
                shape[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_input.png' % b))
            vis.visualize_pointcloud(
                global_rec_shape[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_global_rec.png' % b))
            vis.visualize_pointcloud(
                unfold_pts[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_unfold.png' % b))
            vis.visualize_pointcloud(
                unfold_rec_shape[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_self_rec_shape_u.png' % b))
            vis.visualize_pointcloud(
                cross_rec_shape[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_cross_rec_shapes.png' % b))

            if self.config['training']['mode'] == "stage2":
                self.visualizer.show_pointclouds(points=local_rec_shape_from_unfold[b], title="local_rec_shape_from_unfold_{:02d}".format(b))
                self.visualizer.show_pointclouds(points=local_rec_shape_from_standard[b], title="local_rec_shape_from_standard_{:02d}".format(b))
                vis.visualize_pointcloud(
                    local_rec_shape_from_unfold[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_local_rec_shape_from_unfold.png' % b))
                vis.visualize_pointcloud(
                    local_rec_shape_from_standard[b].data.cpu().numpy(), out_file=os.path.join(self.vis_dir, '%03d_local_rec_shape_from_standard.png' % b))

                self.visualizer.show_sphere_groups(points=batch_p_2d[0], title="val_grouped_sphere", groups=stpts_groups[0], num_groups=stpts_prob_map.size(1))
                self.visualizer.show_sphere_groups(points=stpts_xyz_folded[0], title="val_gs", groups=torch.tensor(np.arange(stpts_prob_map.size(1))), num_groups=stpts_prob_map.size(1))
                self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="val_gs", groups=torch.tensor(np.arange(stpts_prob_map.size(1))), num_groups=stpts_prob_map.size(1))
                self.visualizer.show_sphere_groups(points=local_rec_shape_from_standard[b], title='val_%03d_grouped_shape_local' % b, groups=stpts_groups[b], num_groups=stpts_prob_map.size(1))
                self.visualizer.show_histogram(groups=stpts_groups[0], title="val_histogram", num_groups=stpts_prob_map.size(1))



        return

        # # --------- Encode ---------
        # g_latent = self.model.Encoder(shape)

        # # --------- Reshape to [B,C,H,W] ---------
        # g_latent = g_latent.view(batch_size, self.model.Fold.z_dim, 10, 10)

        # # ---------    VQ   ---------
        # _, g_latent_vq, _, g_latent_vq_indices = self.model.VQ(g_latent) # [B,C,H,W]

        # # --------- get UV pts ---------
        # grid = self.template.get_regular_points(npoints=100*100, level=4)
        # batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)

        # # --------- Interpolate Local Features ---------
        # g_latent_interpolated = F.interpolate(g_latent_vq, scale_factor=10, mode='bilinear', align_corners=True)
        # g_latent_interpolated = g_latent_interpolated.view(batch_size, 16, 100*100).transpose(2,1)  # [B, HxW, C]

        # # --------- Fold with Local Features ---------
        # self_rec_shape = self.model.Fold(g_latent_interpolated, batch_p_2d)      

        # # --------- [Gen] Sample from Transformer ---------
        # c_indices = repeat(torch.tensor([self.model.sos_token]), '1 -> b 1', b=batch_size).to(device)  # sos token
        # index_sample = sample_with_past(c_indices, self.model.Transformer, steps=100)
        
        # # --------- [Gen] Lookup Codebook ---------
        # bhwc = (batch_size,10,10,16)
        # quant_z = self.model.VQ.get_codebook_entry(index_sample.reshape(-1), shape=bhwc)

        # # --------- [Gen] Interpolate Local Features ---------
        # g_latent_interpolated_g = F.interpolate(quant_z, scale_factor=10, mode='bilinear', align_corners=True)
        # g_latent_interpolated_g = g_latent_interpolated_g.view(batch_size, 16, 100*100).transpose(2,1)  # [B, HxW, C]

        # # --------- [Gen] Fold with Local Features ---------
        # gen_shape = self.model.Fold(g_latent_interpolated_g, batch_p_2d)      

        # # Sanity Check
        # bhwc = (batch_size,10,10,16)
        # check_quant_z = self.model.VQ.get_codebook_entry(g_latent_vq_indices.reshape(-1), shape=bhwc)
        # g_latent_interpolated_check = F.interpolate(check_quant_z, scale_factor=10, mode='bilinear', align_corners=True)
        # g_latent_interpolated_check = g_latent_interpolated_check.view(batch_size, 16, 100*100).transpose(2,1)  # [B, HxW, C]
        # rec_shape_check = self.model.Fold(g_latent_interpolated_check, batch_p_2d)      

        # self.visualizer.show_pointclouds(points=shape[0], title="train_input")
        # self.visualizer.show_pointclouds(points=gen_shape[0], title="self_generate_0")
        # self.visualizer.show_pointclouds(points=gen_shape[1], title="self_generate_1")
        # self.visualizer.show_pointclouds(points=gen_shape[2], title="self_generate_2")
        # self.visualizer.show_pointclouds(points=gen_shape[3], title="self_generate_3")

        # for b in trange(batch_size):
        #     input_shape = shape[b].data.cpu().numpy()
        #     rec_shape = self_rec_shape[b].data.cpu().numpy()
        #     sample_shape = gen_shape[b].data.cpu().numpy()
        #     check_shape = rec_shape_check[b].data.cpu().numpy()
        #     input_shape_loc = shape_loc[b].data.cpu().numpy()
        #     input_shape_scale = shape_scale[b].data.cpu().numpy()
        #     input_shape_m = shape_m[b].data.cpu().numpy()
        #     input_shape_s = shape_s[b].data.cpu().numpy()
        #     # structure_pts = l_xyz[4][0:1].squeeze(0).data.cpu().numpy()
        #     # sample_gen_shape = sample_gen_pts[0:1].squeeze(0).data.cpu().numpy()

        #     # --------- de-standardize ---------
        #     # input_shape = (input_shape * input_shape_scale) + input_shape_loc
        #     # rec_shape = (rec_shape * input_shape_scale) + input_shape_loc
        #     # sample_shape = (sample_shape * input_shape_scale) + input_shape_loc
        #     # check_shape = (check_shape * input_shape_scale) + input_shape_loc
        #     # sample_gen_shape = (sample_gen_shape * input_shape_scale) + input_shape_loc

        #     # --------- de-normalize ---------
        #     input_shape = (input_shape * input_shape_s) + input_shape_m
        #     rec_shape = (rec_shape * input_shape_s) + input_shape_m
        #     sample_shape = (sample_shape * input_shape_s) + input_shape_m
        #     check_shape = (check_shape * input_shape_s) + input_shape_m

        #     # logger.add_mesh('%03d_input' % b, vertices=shape[0:1])
        #     # # logger.add_mesh('%03d_unfold' % b, vertices=unfold_pts[0:1])
        #     # logger.add_mesh('%03d_self_rec' % b, vertices=self_rec_pts[0:1])

        #     # --------- vis ---------
        #     vis.visualize_pointcloud(
        #         input_shape, out_file=os.path.join(self.vis_dir, '%03d_input.png' % b))
        #     # vis.visualize_pointcloud(
        #     #     unfold_shape, out_file=os.path.join(self.vis_dir, '%03d_unfold.png' % b))
        #     vis.visualize_pointcloud(
        #         rec_shape, out_file=os.path.join(self.vis_dir, '%03d_self_rec.png' % b))
        #     vis.visualize_pointcloud(
        #         sample_shape, out_file=os.path.join(self.vis_dir, '%03d_sample.png' % b))
        #     vis.visualize_pointcloud(
        #         check_shape, out_file=os.path.join(self.vis_dir, '%03d_check.png' % b))
        #     # vis.visualize_pointcloud(
        #     #     sample_gen_shape, out_file=os.path.join(self.vis_dir, '%03d_generated.png' % b))
        #     # vis.visualize_pointcloud_stp(
        #     #     input_shape, stp_points=structure_pts, out_file=os.path.join(self.vis_dir, '%03d_structure_pts.png' % b))
        
        # vis.visualize_embedding_with_umap(self.model.VQ._embedding.weight.data.cpu(), out_file=os.path.join(self.vis_dir, 'embedding.png'))

        # return 


    def compute_loss(self, data, epoch_it, it):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        self.loss_dict = {}

        # shape = data.get('set').to(device).float()
        shape = data.to(device).float()
        batch_size, n_points = shape.size(0), shape.size(1)

        if self.config['training']['mode'] in ['stage1', 'stage2']:
            # --------- Encode ---------
            g_latent, l_latent = self.model.Encoder(shape)

            # --------- get UV pts ---------
            grid = self.template.get_random_points(torch.Size((1, 3, n_points)))
            batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
            
            # --------- Stack Global Features ---------
            g_latent_stacked = g_latent.view(batch_size, self.model.Fold.z_dim).unsqueeze(1).expand(-1, n_points, -1)

            # --------- [L_rec] ---------
            self_rec_shape = self.model.Fold(g_latent_stacked.detach(), batch_p_2d)
            self.loss_dict['loss_sr'] = CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape)

            # --------- [L_unfold] ---------
            unfold_pts = self.model.Unfold(l_latent, shape)
            self.loss_dict['loss_unfold'] = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d)

            # --------- [L_sr_u] ---------
            self_rec_shape_u = self.model.Fold(g_latent_stacked, unfold_pts)
            # self.loss_dict['loss_sr_u'] = selfrec_loss(self_rec_shape_u, shape) +  CD_loss(self_rec_shape_u, shape) + EMD_loss(self_rec_shape_u, shape)

            # # --------- [L_cr] ---------
            # cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)
            # cross_rec_shapes = self.model.Fold(g_latent_stacked, cross_unfold_pts)
            # self.loss_dict['loss_cr'] = CD_loss(cross_rec_shapes, shape) + EMD_loss(cross_rec_shapes, shape)

            
            if self.config['training']['mode'] == "stage2":
                with torch.no_grad():
                    if self.config['training']['stage2']['normalize_unfold']:
                        f_latent_input = l_latent
                        f_pts_input = (unfold_pts / (unfold_pts.norm(p=2, dim=-1, keepdim=True)+1e-8))
                    elif self.config['training']['stage2']['chamfer_nn']:
                        f_latent_input, shape_order, standard_order = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)
                        f_pts_input = batch_p_2d

                # -------- Grouping -----------
                # with torch.no_grad():
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.model.PG(batch_p_2d, self_rec_shape, f_latent_input)

                # stpts_xyz = (stpts_xyz / (stpts_xyz.norm(p=2, dim=-1, keepdim=True))) / 2
                # stpts_fold_input = torch.cat([stpts_xyz, batch_p_2d], dim=1)
                # g_latent_stpts = g_latent.view(batch_size, self.model.Fold.z_dim).unsqueeze(1).expand(-1, stpts_fold_input.size(1), -1)

                # stpts_xyz_folded = self.model.Fold(g_latent_stpts, stpts_fold_input)[:,:stpts_xyz.size(1),:]
                # self.loss_dict['loss_pg'] = CD_loss(stpts_xyz_folded, shape)
                # self.loss_dict['loss_pg_sphere'] = torch.abs(stpts_xyz.norm(dim=2) - 0.5).sum()

                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    self_rec_shape_f = self.model.Fold_f(scattered_features, f_pts_input) 
                # self.loss_dict['loss_sr_f'] = CD_loss(self_rec_shape_f, shape) + EMD_loss(self_rec_shape_f, shape)

                # _, l_latent_ = self.model.Encoder(self_rec_shape_f)
                # f_latent_input_, _, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent_, device)
                # _, _, _, weighted_features_cycle, _, _ = self.model.PG(batch_p_2d, self_rec_shape, f_latent_input_)
                # self.loss_dict['loss_cycle'] = selfrec_loss(weighted_features_cycle, weighted_features)


                batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())

                # with torch.no_grad():
                stpts_prob_map, stpts_xyz, stpts_groups_, weighted_features_, scattered_features, stpts_xyz_folded_ = self.model.PG(batch_p_2d_in_shape_order, shape, l_latent)
                # self.loss_dict['loss_pg_'] = CD_loss(stpts_xyz_folded_, shape)

                if self.config['training']['stage1']['vq-enabled']:
                    # ---------    [VQ]   ---------
                    VQ_input = scattered_features.permute(0,2,1).view(batch_size, -1, 2, n_points//2)
                    loss_vq, scattered_features, self.perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                    scattered_features= scattered_features.view(batch_size, -1, 2 *(n_points//2)).permute(0,2,1)
                    self.loss_dict['loss_vq'] = loss_vq

                self_rec_shape_f_ = self.model.Fold_f(scattered_features, batch_p_2d_in_shape_order) 
                self.loss_dict['loss_sr_f_'] = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape) # + selfrec_loss(self_rec_shape_f_, shape)


                # _, l_latent_ = self.model.Encoder(self_rec_shape_f_)
                # f_latent_input_, _, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent_, device)
                # _, _, _, weighted_features_cycle, _, _ = self.model.PG(batch_p_2d, self_rec_shape, f_latent_input_)
                # self.loss_dict['loss_cycle'] = selfrec_loss(weighted_features_cycle, weighted_features_)


            if it % 100 == 0:
                self.visualizer.show_pointclouds(points=shape[0], title="train_input")
                self.visualizer.show_pointclouds(points=self_rec_shape[0], title="self_reconstruct")
                # self.visualizer.show_pointclouds(points=unfold_pts[0], title="train_input_unfold")
                self.visualizer.show_pointclouds(points=batch_p_2d[0], title="unfold_target")
                # self.visualizer.show_pointclouds(points=self_rec_shape_u[0], title="self_reconstruct_u")

                if self.config['training']['mode'] == "stage2":
                    self.visualizer.show_sphere_groups(points=shape[0], title="sanity_check", groups=stpts_groups_[0], num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=self_rec_shape_f[0], title="self_rec_shape_f", groups=stpts_groups[0], num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=self_rec_shape_f_[0], title="self_rec_shape_f_", groups=stpts_groups_[0], num_groups=stpts_prob_map.size(1))
                    # self.visualizer.show_pointclouds(points=self_rec_shape_f[0], title="self_rec_shape_f")
                    self.visualizer.show_sphere_groups(points=unfold_pts[0], title="grouped_sphere_", groups=stpts_groups_[0], num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=batch_p_2d[0], title="grouped_sphere", groups=stpts_groups[0], num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=stpts_xyz_folded_[0], title="stpts_xyz_folded", groups=torch.tensor(np.arange(stpts_prob_map.size(1))), num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="gs", groups=torch.tensor(np.arange(stpts_prob_map.size(1))), num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_sphere_groups(points=self_rec_shape[0], title="grouped_shape_global", groups=stpts_groups_[0], num_groups=stpts_prob_map.size(1))
                    self.visualizer.show_histogram(groups=stpts_groups_[0], title="histogram", num_groups=stpts_prob_map.size(1))

                    vis.visualize_embedding_with_umap(self.model.VQ._embedding.weight.data.cpu(), out_file=os.path.join(self.vis_dir, 'embedding.png'))

                    # self.visualizer.show_pointclouds(points=shape_sampled[0], title="shape_sampled")
                    # self.visualizer.show_pointclouds(points=unfold_pts_sampled[0], title="unfold_pts_sampled")

        elif self.config['training']['mode'] == "transformer": 
            _, z_indices = self.model.encode_z(shape) # z_indices = [B, 10x10]
            _, c_indices = self.model.encode_c(shape) # c_indices = [B, 1]
            cz_indices = torch.cat((c_indices, z_indices), dim=1)  
            
            # make the prediction
            logits, _ = self.model.Transformer(cz_indices[:, :-1])

            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, c_indices.shape[1]-1:]

            # transformer prediction loss
            self.loss_dict['loss_transformer'] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))


        # organize loss
        self.loss = 0
        self.loss_dict_mean = OrderedDict()
        for loss_name, loss in self.loss_dict.items():
            single_loss = loss.mean()
            self.loss += single_loss
            self.loss_dict_mean[loss_name] = single_loss.item()

        self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.loss.item()
        
        return self.loss

    def get_current_scalars(self):
        
        sc_dict = OrderedDict([])

        for loss_key in self.loss_dict.keys():
            sc_dict.update({(loss_key, self.loss_dict[loss_key].item())})

        sc_dict.update({
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.loss.item()),
        })

        if self.config['training']['mode'] == "stage2":
            sc_dict.update({
                ('perplexity', self.perplexity.item()),
        })
        # if self.config['training']['mode'] == "vq":
        #     sc_dict = OrderedDict([
        #         # ('loss_unfold', self.loss_dict['loss_unfold'].item()),
        #         ('loss_sr', self.loss_dict['loss_sr'].item()),
        #         # ('loss_kld', self.loss_dict['loss_kld'].item()),
        #         # ('loss_vq', self.loss_dict['loss_vq'].item()),
        #         # ('perplexity', self.perplexity.item()),
        #         # ('loss_cr', self.loss_dict['loss_cr'].item()),
        #         ('smoothed_total_loss', self.smoothed_total_loss),
        #         ('total_loss', self.loss.item()),
        #     ])
        #     if self.config['model']['latent'] == "local": 
        #         sc_dict.update({
        #             ('loss_unfold', self.loss_dict['loss_unfold'].item()),
        #             ('loss_sr_u', self.loss_dict['loss_sr_u'].item()),
        #             ('loss_cr', self.loss_dict['loss_cr'].item()),
        #             ('loss_sr_f', self.loss_dict['loss_sr_f'].item()),

        #         })
        # elif self.config['training']['mode'] == "transformer": 
        #     sc_dict = OrderedDict([
        #         ('loss_transformer', self.loss_dict['loss_transformer'].item()),
        #         ('smoothed_total_loss', self.smoothed_total_loss),
        #         ('total_loss', self.loss.item()),
        #     ])
        
        return sc_dict