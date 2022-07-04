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
        self.optimizers = self.model.configure_optimizers(config)
        self.device = device
        self.vis_dir = vis_dir
        self.eval_sample = eval_sample
        self.smoothed_total_loss = 0
        self.config = config
        self.template = SphereTemplate(device=device)
        self.start_visdom()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, epoch_it, it):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        for opt in self.optimizers:
            opt.zero_grad()
        loss = self.compute_loss(data, epoch_it, it)
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        return loss.item()

    def train_step_d(self, data, epoch_it, it):
        self.model.eval()
        self.model.D.train()
        self.optimizerD.zero_grad()
        loss = self.compute_loss_d(data, epoch_it, it)
        loss.backward()
        self.optimizerD.step()
        return loss.item()

    def eval_step(self, data, epoch_it, save=None):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''

        device = self.device
        self.model.eval()

        shape = data.get('pointcloud').to(self.device).float()
        shape_loc, shape_scale  = data.get('shift').to(device).float(), data.get('scale').to(device).float()
        # shape_m, shape_s = data.get('mean').to(device).float(), data.get('std').to(device).float()

        batch_size, n_points = shape.size(0), shape.size(1)

        with torch.no_grad():
            g_latent, l_latent = self.model.Encoder(shape)

        # grid = self.template.get_regular_points(level=4)
        grid = self.template.get_random_points(torch.Size((1, 3, 2048)))
        batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
        g_latent_stacked = g_latent.unsqueeze(1).expand(-1, grid.size(0), -1)

        with torch.no_grad():
            global_rec_shape = self.model.Fold(g_latent_stacked, batch_p_2d)
        with torch.no_grad():
            unfold_pts = self.model.Unfold(l_latent, shape)

        if self.config['training']['mode'] == "stage2":
            f_latent_input, shape_order, standard_order = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)

            f_latent_input, _, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)

            with torch.no_grad():
                # from standard
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.model.PG(batch_p_2d, global_rec_shape, f_latent_input)

                """
                VQ_input = scattered_features.permute(0,2,1) # .view(batch_size, -1, 2, n_points//2)
                loss_vq, scattered_features, perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                scattered_features= scattered_features.view(batch_size, -1, n_points).permute(0,2,1)
                """

                local_rec_shape_from_standard = self.model.Fold_f(scattered_features, batch_p_2d.detach()) 

                # from unfold
                batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())
                _, _, stpts_groups_, _, scattered_features, stpts_xyz_folded_ = self.model.PG(batch_p_2d_in_shape_order, shape, l_latent)
                
                VQ_input = scattered_features.permute(0,2,1) # .view(batch_size, -1, 2, n_points//2)
                loss_vq, scattered_features, perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                scattered_features= scattered_features.view(batch_size, -1, 2 *(n_points//2)).permute(0,2,1)

                local_rec_shape_from_unfold = self.model.Fold_f(scattered_features, batch_p_2d_in_shape_order) 

        if self.config['training']['mode'] == "stage3":
            key_length = 120
            with torch.no_grad():
                batch_size = batch_size
                # --------- [Gen] Transformer Sample Sequence ---------
                
                c_indices = repeat(torch.tensor([self.model.sos_token]), '1 -> b 1', b=batch_size).to(device)  # sos token
                index_sample = sample_with_past(c_indices, self.model.Transformer, steps=key_length)
                
                # --------- [Gen] Query VQ Codebook ---------
                quant_z_size = (batch_size, key_length, 128)

                # fp = open("/home/vslab2018/3d/CanonicalVAE/out/default/stage3/default_dev/gen_all.txt", "a")
                # fp.write('train: '+''.join(str(list(index_sample.cpu().numpy())))+'\n')

                # fp = open("/home/vslab2018/3d/CanonicalVAE/out/default/stage3/default_dev/gen_unique.txt", "a")
                # fp.write('train: '+''.join(str(list(index_sample.unique().cpu().numpy())))+'\n')

                # index_sample = index_sample *  0 + 4743

                quant_z = self.model.VQ.get_codebook_entry(index_sample.reshape(-1), shape=None)
                # import pdb; pdb.set_trace()
                quant_z = quant_z.view(quant_z_size)


                # --------- [Gen] Fold with Local Features ---------
                batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
                stpts_groups_ = self.model.PG.get_groups(batch_p_2d)
                propagated_features = self.model.PG.propagate_feat(batch_p_2d, quant_z)
                gen_pts = self.model.Fold_f(propagated_features, batch_p_2d)     
                self.visualizer.show_sphere_groups(points=gen_pts[0], title="4743_{:02d}".format(0), groups=stpts_groups_[0], num_groups=256)

                for b in range(batch_size):
                    self.visualizer.show_sphere_groups(points=gen_pts[b], title="eval_gen_pts_{:02d}".format(b), groups=stpts_groups_[b], num_groups=256)

        if self.config['training']['mode'] == "stage3":
            gdt_shape = shape
            gen_shape = gen_pts
            # gen_shape = denormalize(gen_pts, shape_loc, shape_scale)
            # gdt_shape = denormalize(shape, shape_loc, shape_scale)
            rt = {
                'gdt_shape': gdt_shape,
                'gen_shape': gen_shape,
                # 'rec_shape': rec_shape,
            }
        elif self.config['training']['mode'] == "stage2":
            local_rec_shape_from_standard = denormalize(local_rec_shape_from_standard, shape_loc, shape_scale)
            local_rec_shape_from_unfold = denormalize(local_rec_shape_from_unfold, shape_loc, shape_scale)
            global_rec_shape = denormalize(global_rec_shape, shape_loc, shape_scale)
            gdt_shape = denormalize(shape, shape_loc, shape_scale)
            rt = {
                'gdt_shape': gdt_shape,
                'rec_shape': local_rec_shape_from_standard,
                'local_rec_shape_from_unfold': local_rec_shape_from_unfold,
                'global_rec_shape': global_rec_shape,
            }
        else:
            gdt_shape = denormalize(shape, shape_loc, shape_scale)
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


    def compute_loss(self, data, epoch_it, it):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        self.loss_dict = {}

        # shape = data.get('pointcloud').to(device).float()
        shape = data.to(device).float()
        batch_size, n_points, num_groups = shape.size(0), shape.size(1), self.model.PG.num_groups

        if self.config['training']['mode'] == "stage3": 
            # _, z_indices = self.model.encode_z(shape) # z_indices = [B, 10x10]
            # _, c_indices = self.model.encode_c(shape) # c_indices = [B, 1]
            # cz_indices = torch.cat((c_indices, z_indices), dim=1)  
            with torch.no_grad():
                g_latent, l_latent = self.model.Encoder(shape)
                grid = self.template.get_random_points(torch.Size((1, 3, n_points)))
                batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
                unfold_pts = self.model.Unfold(l_latent, shape)
                f_latent_input, shape_order, _ = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)
                batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())
                stpts_prob_map, stpts_xyz, stpts_groups_, weighted_features_, scattered_features, stpts_xyz_folded_ = self.model.PG(batch_p_2d_in_shape_order, shape, l_latent)
                weighted_features_ = self.model.PG.reorder_weighted_features(weighted_features_)
                
                VQ_input = weighted_features_.permute(0,2,1)
                _, _, _, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]

                z_indices = g_latent_vq_index.view(shape.size(0),-1)
            
            _, c_indices = self.model.encode_c(shape) # c_indices = [B, 1]
            cz_indices = torch.cat((c_indices, z_indices), dim=1)  
            
            # make the prediction
            logits, _ = self.model.Transformer(cz_indices[:, :-1])

            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, c_indices.shape[1]-1:]

            # transformer prediction loss
            self.loss_dict['loss_transformer'] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))

            if it % 50 == 0:
                self.model.eval()
                # --------- [Gen] Transformer Sample Sequence ---------
                c_indices = repeat(torch.tensor([self.model.sos_token]), '1 -> b 1', b=3).to(device)  # sos token
                index_sample = sample_with_past(c_indices, self.model.Transformer, steps=weighted_features_.size(1))
                
                # --------- [Gen] Query VQ Codebook ---------
                quant_z_size = (3, weighted_features_.size(1), 128)
                quant_z = self.model.VQ.get_codebook_entry(index_sample.reshape(-1), shape=None)
                quant_z = quant_z.view(quant_z_size)

                # --------- [Gen] Fold with Local Features ---------
                stpts_groups_ = self.model.PG.get_groups(batch_p_2d[0:3,:,:])
                propagated_features = self.model.PG.propagate_feat(batch_p_2d[0:3,:,:], quant_z)
                gen_pts = self.model.Fold_f(propagated_features, batch_p_2d[0:3,:,:])     
                self.model.train()

                self.visualizer.show_sphere_groups(points=stpts_xyz_folded_[0], title="stpts_xyz_folded", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="stpts_xyz", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=shape[0], title="grouped_shape", groups=stpts_groups_[0], num_groups=num_groups)
                # self.visualizer.show_sphere_groups(points=batch_p_2d[0], title="grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_histogram(groups=stpts_groups_[0], title="histogram", num_groups=num_groups)

                # self.visualizer.show_sphere_groups(points=gen_pts[0], title="gen_pts_0", groups=stpts_groups_[0], num_groups=256)
                # self.visualizer.show_sphere_groups(points=gen_pts[1], title="gen_pts_1", groups=stpts_groups_[1], num_groups=256)
                # self.visualizer.show_sphere_groups(points=gen_pts[2], title="gen_pts_2", groups=stpts_groups_[2], num_groups=256)


        if self.config['training']['mode'] in ['stage1', 'stage2']:
            # --------- Encode ---------
            g_latent, l_latent = self.model.Encoder(shape)

            # --------- get UV pts ---------
            grid = self.template.get_random_points(torch.Size((1, 3, n_points)))
            batch_p_2d = self.template.get_random_points(torch.Size((batch_size, 3, n_points))).permute(1,2,0)
            # batch_p_2d = batch_sample_from_grid(grid, batch_size, downsample=False)
            
            # --------- Stack Global Features ---------
            g_latent_stacked = g_latent.view(batch_size, self.model.Fold.z_dim).unsqueeze(1).expand(-1, n_points, -1)

            # --------- [L_rec] ---------
            self_rec_shape = self.model.Fold(g_latent_stacked, batch_p_2d)
            self.loss_dict['loss_sr'] = CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape) # V

            # --------- [L_unfold] ---------
            unfold_pts = self.model.Unfold(g_latent_stacked, shape) # l_latent g_latent_stacked
            self.loss_dict['loss_unfold'] = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d) # V

            # --------- [L_sr_u] ---------
            self_rec_shape_u = self.model.Fold(g_latent_stacked, unfold_pts)
            self.loss_dict['loss_sr_u'] = selfrec_loss(self_rec_shape_u, shape) +  CD_loss(self_rec_shape_u, shape) + EMD_loss(self_rec_shape_u, shape)
            
            # if self.config['training']['mode'] in ['stage1']:
            #     # --------- [L_local] ---------
            #     self_rec_shape_f = self.model.Fold_f(l_latent, unfold_pts.detach()) 
            #     self.loss_dict['loss_sr_f'] = CD_loss(self_rec_shape_f, shape)

            # # # --------- [L_cr] ---------
            # cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)
            # cross_rec_shapes = self.model.Fold(g_latent_stacked.detach(), cross_unfold_pts)
            # self.loss_dict['loss_cr'] = CD_loss(cross_rec_shapes, shape) + EMD_loss(cross_rec_shapes, shape)

            # -------- generate orders -----------
            f_latent_input, shape_order, standard_order = get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent, device)
            batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())

            # -------- [L_PG] ----------- 
            if self.config['training']['mode'] in ["stage1", "stage2"]: # , "stage2"
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.model.PG(batch_p_2d_in_shape_order.detach(), shape.detach(), l_latent)
                self.loss_dict['loss_pg'] = CD_loss(stpts_xyz_folded, shape) # + EMD_loss (stpts_xyz_folded, sub_shape)
                self.num_unique_groups = self.model.PG.num_unique_groups
            else:
                stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.model.PG(batch_p_2d_in_shape_order, shape, l_latent)
                self.pg_loss = CD_loss(stpts_xyz_folded, shape)
                self.num_unique_groups = self.model.PG.num_unique_groups

            if self.config['training']['mode'] == "stage2":

                # VQ_input = scattered_features.permute(0,2,1) # .view(batch_size, -1, 2, n_points//2)
                # loss_vq, scattered_features, self.perplexity, g_latent_vq_index = self.model.VQ(VQ_input) # [B,C,H,W]
                # scattered_features= scattered_features.view(batch_size, -1, n_points).permute(0,2,1)
                # self.loss_dict['loss_vq'] = loss_vq

                with torch.no_grad():
                    self_rec_shape_f_ = self.model.Fold_f(scattered_features, batch_p_2d_in_shape_order) 
                # self.loss_dict['loss_sr_f_'] = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape) # + selfrec_loss(self_rec_shape_f_, shape)
                pass
            
            if it % 100 == 0:
                self.visualizer.show_pointclouds(points=shape[0], title="train_input")
                self.visualizer.show_pointclouds(points=self_rec_shape[0], title="self_reconstruct")
                self.visualizer.show_pointclouds(points=self_rec_shape_u[0], title="self_reconstruct_u")
                self.visualizer.show_pointclouds(points=unfold_pts[0], title="train_input_unfold")
                self.visualizer.show_pointclouds(points=batch_p_2d[0], title="unfold_target")

                self.visualizer.show_sphere_groups(points=stpts_xyz_folded[0], title="stpts_xyz_folded", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="stpts_xyz", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=shape[0], title="grouped_shape", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=unfold_pts[0], title="grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                # self.visualizer.show_sphere_groups(points=batch_p_2d[0], title="grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_histogram(groups=stpts_groups[0], title="histogram", num_groups=num_groups)
                
                # self.visualizer.show_pointclouds(points=self_rec_shape_u[0], title="self_reconstruct_u")

                if self.config['training']['mode'] == "stage2":
                    self.visualizer.show_sphere_groups(points=self_rec_shape_f_[0], title="self_rec_shape_f_", groups=stpts_groups[0], num_groups=num_groups)


                    """
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
                    """
                    # vis.visualize_embedding_with_umap(self.model.VQ._embedding.weight.data.cpu(), out_file=os.path.join(self.vis_dir, 'embedding.png'))

                    # self.visualizer.show_pointclouds(points=shape_sampled[0], title="shape_sampled")
                    # self.visualizer.show_pointclouds(points=unfold_pts_sampled[0], title="unfold_pts_sampled")

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
    
        if self.config['training']['mode'] in ["stage1", "stage2"]:
            sc_dict.update({
                ('num_unique_groups', self.num_unique_groups),
        })
        
        # if self.config['training']['mode'] == "stage2":
        #     sc_dict.update({
        #         ('perplexity', self.perplexity.item()),
        #         # ('pg_loss_static', self.pg_loss.item()),
        # })
        
        return sc_dict