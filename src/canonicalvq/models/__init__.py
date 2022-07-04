import torch
import torch.nn as nn
import torch.optim as optim

from src.canonicalvq.models import encoder, decoder, transformer
from src.utils.common import SOSProvider
from src.utils.template import SphereTemplate
from src.canonicalvq.loss import *
from external.metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD, emd_cd
from src.utils.visualize import Visualizer
from torch.distributions.categorical import Categorical
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from src.canonicalvq.models import vector_quantize

from einops import repeat
from src.canonicalvq.models.transformer.mingpt import sample_with_past

import pytorch_lightning as pl
import time


__all__ = [
    encoder, decoder, transformer,
]

# not supposely to be used
# Decoder dictionary
decoder_dict = {
    'ImplicitFun': decoder.ImplicitFun,
    'FixPointGenerator': decoder.FixPointGenerator,
    'SPGANGenerator': decoder.SPGANGenerator,
    'PrimitiveGrouping': decoder.PrimitiveGrouping,
}

encoder_dict = {
    'PointNetfeat': encoder.PointNetfeat,
    'DGCNN': encoder.DGCNN,
}

class CanonicalVQ(pl.LightningModule):
    def __init__(self,
                 feat_dim,
                 n_embed,
                 num_groups,
                 codebook_size=100,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 mode=1,
                 train_unfold=True,
                 no_fold=False,
                 sane_index_shape=False,
                 repro_atlas=False,
                 pg_from_standard=False,
                 condition=None,
                 learnable=True,
                 query_points=5000,
                 codebook_dim=4,
                 abl=None,
                 ):
        super().__init__()

        self.mode = mode

        self.feat_dim = feat_dim
        self.num_groups = num_groups

        feat_dim_local = feat_dim
        self.feat_dim_local = feat_dim_local
        self.train_unfold = train_unfold
        self.no_fold = no_fold
        self.pg_from_standard = pg_from_standard
        self.learnable = learnable
        self.query_points = query_points

        self.template = SphereTemplate()
        self.level = 3

        self.condition = condition

        self.Encoder = encoder.DGCNN(feat_dim=feat_dim)
        self.Fold = decoder.SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=False) # local

        self.Encoder_f = encoder.DGCNN(feat_dim=feat_dim)
        # self.Unfold = decoder.SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True)
        self.Fold_f = decoder.SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=False) # local
        self.PG = decoder.PrimitiveGrouping(num_groups=num_groups, learnable=learnable, abl=abl)
        self.codebook_size = codebook_size
        self.VQ = vector_quantize.GroupedVQ(num_groups, codebook_size=codebook_size, codebook_dim=codebook_dim)

        self.temperature_scheduler = None

        self.ae_loss = CanonicalVQLoss()
        self.correspondence_loss = CorrespondenceLoss()

        # self.quant_conv = torch.nn.Conv1d(feat_dim, feat_dim_local, 1)
        # self.post_quant_conv = torch.nn.Conv1d(feat_dim_local, feat_dim, 1)

        if self.mode == 3:
            # if self.condition == 'mask':
            #     import torchvision.models as models
            #     # init a pretrained resnet
            #     backbone = models.resnet50(pretrained=True)
            #     num_filters = backbone.fc.in_features
            #     layers = list(backbone.children())[:-1]
            #     self.cond_feature_extractor = nn.Sequential(*layers)
            #     self.cond_feature_projection = nn.Linear(num_filters, 256)
            #     cond_feature_dim = 256
            # else:
            cond_feature_dim = 0

            self.Encoder.eval()
            self.Fold.eval()
            # self.Unfold.eval()
            self.PG.eval()
            self.VQ.eval()
            # self.quant_conv.eval()
            # self.post_quant_conv.eval()
            
            self.Transformer = transformer.mingpt.GPT(vocab_size=codebook_size, 
                                                    block_size=num_groups,
                                                    n_layer=24, # 24
                                                    n_head=16, # 16
                                                    n_embd=256, # 512
                                                    cond_feature_dim=cond_feature_dim) 
            self.be_unconditional = True
            # self.cond_stage_key = self.first_stage_key
            self.sos_token = 0
            self.cond_model = SOSProvider(self.sos_token)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.image_key = image_key
        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels)==int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.visualizer = None

    def start_visdom(self, env='default', port=5000):
        self.visualizer = Visualizer(port, env)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, batch, denorm=False):
        shape = batch.get('pointcloud').float() #tr
        shape_tr = batch.get('pointcloud').float() #tr
        shape_te = batch.get('pointcloud_ref').float() #te
        # shape_random = batch.get('pointcloud_random').float()

        shape_loc, shape_scale  = batch.get('shift').float(), batch.get('scale').float()
        batch_size, n_points = shape.size(0), shape.size(1)

        with torch.no_grad():
            corr_out = self.forward_correspondence(shape, n_points=2048)

        g_latent = corr_out["g_latent"]
        l_latent = corr_out["l_latent"]
        batch_p_2d = corr_out["batch_p_2d"]
        global_rec_shape = corr_out["self_rec_shape"]
        unfold_pts = corr_out["unfold_pts"]
        batch_p_2d_in_shape_order = corr_out["batch_p_2d_in_shape_order"]
        l_latent_in_standard_order = corr_out["l_latent_in_standard_order"]

        # [from standard]
        # -------- PG -----------
        self.PG.regular_points = self.template.get_regular_points(level=self.level)
        _, _, _, grouped_features, scattered_features, _ = self.PG(batch_p_2d, global_rec_shape, l_latent_in_standard_order)
        # scattered_features = corr_out['g_latent_stacked']
        # -------- VQ -----------

        # scattered_features = self.quant_conv(scattered_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # grouped_features = self.quant_conv(grouped_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        grouped_quant, indices, _ = self.VQ(grouped_features) # grouped_quant, indices, emb_loss
        # grouped_quant = self.post_quant_conv(grouped_quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        quant = self.PG.remap_grouped_features(batch_p_2d, grouped_quant)
        # quant = self.PG.remap_grouped_features(batch_p_2d, grouped_features)
        # quant = self.post_quant_conv(quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # -------- Fold_f -----------
        # quant = corr_out['g_latent_stacked']
        # quant = scattered_features
        self_rec_shape_f = self.Fold_f(quant, batch_p_2d)

        # --------- [SANITY] Query VQ Codebook ---------
        quant_z_size = (batch_size, self.num_groups, self.feat_dim_local)
        grouped_features = self.VQ.get_codebook_entry(indices, shape=quant_z_size, reorder_by=None) # 16, 231, 256

        # --------- [SANITY] Fold with Local Features ---------
        quant = self.PG.remap_grouped_features(batch_p_2d, grouped_quant)
        sanity_rec_pts = self.Fold_f(quant, batch_p_2d)     

        # [from unfold]
        # -------- PG -----------
        stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.PG(batch_p_2d_in_shape_order, shape, l_latent)
        _, _, _, grouped_features, scattered_features, _ = self.PG(batch_p_2d_in_shape_order, shape, l_latent)
        # scattered_features = corr_out['g_latent_stacked']
        # -------- VQ -----------

        # scattered_features = self.quant_conv(scattered_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # grouped_features = self.quant_conv(grouped_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        grouped_quant, _, _ = self.VQ(grouped_features)
        # grouped_quant = self.post_quant_conv(grouped_quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        
        quant = self.PG.remap_grouped_features(batch_p_2d_in_shape_order, grouped_quant)
        # quant = self.PG.remap_grouped_features(batch_p_2d, grouped_features)
        # quant = self.post_quant_conv(quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # -------- Fold_f -----------
        # quant = corr_out['g_latent_stacked']
        # quant = scattered_features
        self_rec_shape_f_ = self.Fold_f(quant, batch_p_2d_in_shape_order)

        if denorm:
            shape_te_denormed = denormalize(shape_te, shape_loc, shape_scale)
            self_rec_shape_f = denormalize(self_rec_shape_f, shape_loc, shape_scale)
            self_rec_shape_f_ = denormalize(self_rec_shape_f_, shape_loc, shape_scale)
            global_rec_shape = denormalize(global_rec_shape, shape_loc, shape_scale)
            oracle_shape = denormalize(shape_tr, shape_loc, shape_scale)
            # gen_shape = denormalize(gen_shape, shape_loc, shape_scale)
            sanity_rec_shape = denormalize(sanity_rec_pts, shape_loc, shape_scale)
            
        info = {
            'oracle_shape': oracle_shape,
            'stpts_xyz_folded': stpts_xyz_folded,
            'stpts_xyz': stpts_xyz,
            'unfold_pts': unfold_pts,
            'stpts_groups': stpts_groups,
            'vq_indices': indices,
        }

        rt = {
            'gdt_shape': shape_te_denormed,
            'rec_shape': self_rec_shape_f,
            'local_rec_shape_from_unfold': self_rec_shape_f_,
            'global_rec_shape': global_rec_shape,
            'oracle_shape': oracle_shape,
            'sanity_rec_shape': sanity_rec_shape,
            'info': info,
        }

        if self.mode == 3:
            key_length = self.PG.num_unique_groups

            if self.condition ==  None:
                with torch.no_grad():
                    # --------- [Gen] Transformer Sample Sequence ---------

                    tf_start = time.time()
                    
                    c_indices = repeat(torch.tensor([self.sos_token]), '1 -> b 1', b=batch_size).to(shape).long() # sos token
                    index_sample = sample_with_past(c_indices, self.Transformer, steps=key_length, top_k=50, top_p=0.92, embeddings=None) # , top_k=50, top_p=0.5

                    tf_time = time.time() - tf_start
                    # --------- [Gen] Query VQ Codebook ---------
                    vq_start = time.time()

                    quant_z_size = (batch_size, key_length, self.feat_dim_local)
                    quant_z = self.VQ.get_codebook_entry(index_sample, shape=quant_z_size, reorder_by=self.PG.fibonacci_grid_groups_unique) # 16, 231, 256

                    # --------- [Gen] Fold with Local Features ---------
                    stpts_groups = self.PG.get_groups(batch_p_2d)
                    propagated_features = self.PG.propagate_feat(batch_p_2d, quant_z) # [16, 2048, 256]

                    # post_quant_features = self.post_quant_conv(propagated_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                    gen_pts = self.Fold_f(propagated_features, batch_p_2d)    

                    vq_time = time.time() - vq_start 
            
                gen_info = {
                    'stpts_groups': stpts_groups,
                    'tf_time': tf_time,
                    'vq_time': vq_time,
                }

                rt.update({
                    'gen': gen_info
                })

                rt. update({
                    'gen_ref_shape': shape_te,
                    'gen_our_shape': gen_pts,
                })

            elif self.condition == 'mask':
                image = batch.get('image').float().view(-1, 3, 480, 480)
                num_of_views = image.size(0) // batch_size
                cond_feature = self.cond_feature_extractor(image).flatten(1)
                cond_feature = self.cond_feature_projection(cond_feature)

                gen_pts_denormed_list = []
                gen_pts_list = []

                for k in range(5):

                    with torch.no_grad():
                        # --------- [Gen] Transformer Sample Sequence ---------
                        
                        c_indices = repeat(torch.tensor([self.sos_token]), '1 -> b 1', b=batch_size*num_of_views).to(shape).long() # sos token
                        index_sample = sample_with_past(c_indices, self.Transformer, steps=key_length, temperature=2, top_k=50, top_p=0.92, embeddings=cond_feature)

                        # --------- [Gen] Query VQ Codebook ---------
                        quant_z_size = (batch_size*num_of_views, key_length, self.feat_dim_local)
                        quant_z = self.VQ.get_codebook_entry(index_sample, shape=quant_z_size, reorder_by=self.PG.fibonacci_grid_groups_unique) # 16, 231, 256

                        # --------- [Gen] Fold with Local Features ---------
                        expanded_batch_p_2d = batch_p_2d.unsqueeze(1).expand(-1, num_of_views, -1, -1).reshape(batch_size*num_of_views, 2048, 3)
                        stpts_groups = self.PG.get_groups(expanded_batch_p_2d)
                        propagated_features = self.PG.propagate_feat(expanded_batch_p_2d, quant_z) # [16, 2048, 256]

                        # post_quant_features = self.post_quant_conv(propagated_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                        gen_pts = self.Fold_f(propagated_features, expanded_batch_p_2d)     
                
                    gen_info = {
                        'stpts_groups': stpts_groups,
                    }

                    shape_loc_expanded = shape_loc.unsqueeze(1).expand(-1, num_of_views, -1, -1).reshape(batch_size*num_of_views, 1, 3)
                    shape_scale_expanded = shape_scale.unsqueeze(1).expand(-1, num_of_views, -1, -1).reshape(batch_size*num_of_views, 1, 1)

                    gen_pts_list.append(gen_pts.view(batch_size, num_of_views, 2048, 3)[:,:, None,:,:])

                    gen_pts_denormed = denormalize(gen_pts, shape_loc_expanded, shape_scale_expanded).view(batch_size, num_of_views, 2048, 3)
                    gen_pts_denormed_list.append(gen_pts_denormed[:,:, None,:,:])

                gen_pts_denormed_list = torch.cat(gen_pts_denormed_list, dim=2)
                gen_pts_list = torch.cat(gen_pts_list, dim=2)

                gen_info = {
                    'stpts_groups': stpts_groups,
                }

                gen_info.update({
                    'completion_image': image.view(batch_size, -1, 3, 480, 480).cpu().detach().numpy(),
                })

                rt.update({
                    'gen': gen_info
                })

                rt. update({
                    'completion_ref_shape': shape_te_denormed.unsqueeze(1).expand(-1, num_of_views, -1, -1).cpu().detach().numpy(),
                    'completion_gen_shape': gen_pts_denormed_list.cpu().detach().numpy(),
                    'completion_gen_shape_norm': gen_pts_list.cpu().detach().numpy(),
                })

        return rt

    def generate_samples(self, batch_size=10, DEVICE=None, n_points=2048, query_points=5000):
        batch_p_2d = self.template.get_random_points(torch.Size((1, 3, n_points))).unsqueeze(0)
        batch_p_2d = batch_p_2d.expand(batch_size, -1, -1)
        
        self.PG.get_fibonacci_grid_groups_unique(batch_p_2d, query_points=query_points)
        key_length = self.PG.num_unique_groups

        with torch.no_grad():
            # --------- [Gen] Transformer Sample Sequence ---------
            
            c_indices = repeat(torch.tensor([self.sos_token]), '1 -> b 1', b=batch_size).to(batch_p_2d).long() # sos token
            index_sample = sample_with_past(c_indices, self.Transformer, steps=key_length, top_k=40, top_p=0.92) # , top_k=50, top_p=0.5

            # --------- [Gen] Query VQ Codebook ---------
            quant_z_size = (batch_size, key_length, self.feat_dim_local)
            quant_z = self.VQ.get_codebook_entry(index_sample, shape=quant_z_size, reorder_by=self.PG.fibonacci_grid_groups_unique) # 16, 231, 256

            # --------- [Gen] Fold with Local Features ---------
            stpts_groups = self.PG.get_groups(batch_p_2d)
            propagated_features = self.PG.propagate_feat(batch_p_2d, quant_z) # [16, 2048, 256]

            # post_quant_features = self.post_quant_conv(propagated_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            gen_pts = self.Fold_f(propagated_features, batch_p_2d)
        
        rt = {
            'gen_pts': gen_pts,
            'stpts_groups': stpts_groups,
            'group_order': self.PG.fibonacci_grid_groups_unique.long(),
            'batch_p_2d': batch_p_2d,
        }

        return rt


    def temperature_scheduling(self):
        if self.temperature_scheduler != None:
            self.VQ.temperature = self.temperature_scheduler(self.global_step)
            self.log("train/temperature", self.VQ.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def forward_correspondence(self, shape, fast=False, n_points=None):
        if n_points is None:
            n_points = shape.size(1)
        batch_size, num_groups = shape.size(0), self.PG.num_groups

        # --------- encode features ---------
        g_latent, l_latent = self.Encoder(shape)

        if self.Encoder_f is not None:
            _, l_latent = self.Encoder_f(shape)

        # --------- get UV pts ---------
        batch_p_2d = self.template.get_random_points(torch.Size((batch_size, 3, n_points))).permute(1,2,0).contiguous()

        # --------- Stack Global Features ---------
        g_latent_stacked = g_latent.view(batch_size, self.Fold.z_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)

        # --------- [L_rec] ---------
        self_rec_shape = self.Fold(g_latent_stacked, batch_p_2d)

        if fast:
            out = {
                "g_latent": g_latent,
                "l_latent": l_latent,
                "batch_p_2d": batch_p_2d,
                "g_latent_stacked": g_latent_stacked,
                "self_rec_shape": self_rec_shape,
                "unfold_pts": batch_p_2d, #
                "self_rec_shape_u": batch_p_2d, #
                "shape_order": None, #
                "standard_order": None, #
                "batch_p_2d_in_shape_order": batch_p_2d, #
                "l_latent_in_standard_order": l_latent, #
            }
            return out

        if not self.train_unfold:
            # -------- generate orders -----------
            shape_order, standard_order = get_order_by_chamfer_index(shape, self_rec_shape)
            batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())
            l_latent_in_standard_order = batched_index_select(l_latent, 1, standard_order.long())
            shape_in_standard_order = batched_index_select(shape, 1, standard_order.long())
            unfold_pts = batch_p_2d_in_shape_order
            self_rec_shape_u = batch_p_2d

        else:
            # --------- [L_unfold] ---------
            unfold_pts = self.Unfold(g_latent_stacked, shape) # l_latent g_latent_stacked

            # --------- [L_sr_u] ---------
            self_rec_shape_u = self.Fold(g_latent_stacked, unfold_pts)

            # -------- generate orders -----------
            shape_order, standard_order = get_order_by_chamfer_index(unfold_pts, batch_p_2d)
            batch_p_2d_in_shape_order = batched_index_select(batch_p_2d, 1, shape_order.long())
            l_latent_in_standard_order = batched_index_select(l_latent, 1, standard_order.long())

        out = {
            "g_latent": g_latent,
            "l_latent": l_latent,
            "batch_p_2d": batch_p_2d,
            "g_latent_stacked": g_latent_stacked,
            "self_rec_shape": self_rec_shape,
            "unfold_pts": unfold_pts,
            "self_rec_shape_u": self_rec_shape_u,
            "shape_order": shape_order,
            "standard_order": standard_order,
            "batch_p_2d_in_shape_order": batch_p_2d_in_shape_order,
            "l_latent_in_standard_order": l_latent_in_standard_order,
            "shape_in_standard_order": shape_in_standard_order,
        }

        return out

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # rank = torch.distributed.get_rank()
        if self.temperature_scheduler != None:
            self.temperature_scheduling()

        optimizer_idx = self.mode

        shape = batch.get('pointcloud').float()
        batch_size, n_points, num_groups = shape.size(0), shape.size(1), self.PG.num_groups
        if batch_size <= 1:
            return None

        if optimizer_idx == 0:
            corr_out = self.forward_correspondence(shape, fast=True)
            corr_loss, log_dict_corr = self.correspondence_loss(shape, 
                                                                corr_out['batch_p_2d'], 
                                                                corr_out['self_rec_shape'], 
                                                                corr_out['unfold_pts'], 
                                                                corr_out['self_rec_shape_u'],
                                                                train_unfold=self.train_unfold,
                                                                no_fold=self.no_fold)

            self.log("train/corr_loss", corr_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True) 
            self.log_dict(log_dict_corr, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            if batch_idx % 20 == 0 and self.visualizer != None:
                self.visualizer.show_pointclouds(points=shape[0], title="train/input")
                self.visualizer.show_pointclouds(points=corr_out['self_rec_shape'][0], title="train/global_reconstruct")
                self.visualizer.show_pointclouds(points=corr_out['self_rec_shape_u'][0], title="train/global_reconstruct_u")
                self.visualizer.show_pointclouds(points=corr_out['unfold_pts'][0], title="train/global_unfold")
                self.visualizer.show_pointclouds(points=corr_out['batch_p_2d'][0], title="train/unfold_target")

                # self.visualizer.show_pointclouds(points=self_rec_shape_f_[0], title="train/self_rec_shape_f_")
                # self.visualizer.show_sphere_groups(points=stpts_xyz_folded[0], title="train/stpts_xyz_folded", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                # self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="train/stpts_xyz", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                # self.visualizer.show_sphere_groups(points=shape[0], title="train/grouped_shape", groups=stpts_groups[0], num_groups=num_groups)
                # self.visualizer.show_sphere_groups(points=unfold_pts[0], title="train/grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                # self.visualizer.show_histogram(groups=stpts_groups[0], title="train/histogram", num_groups=num_groups)
            return corr_loss

        if optimizer_idx == 1:
            with torch.no_grad():
                corr_out = self.forward_correspondence(shape)

            if self.pg_from_standard:
                stpts_prob_map, stpts_xyz, stpts_groups, _, scattered_features, stpts_xyz_folded = self.PG(corr_out['batch_p_2d'].detach(), 
                                                                                                        corr_out['shape_in_standard_order'].detach(), 
                                                                                                        corr_out['l_latent_in_standard_order'].detach())
            else:
                stpts_prob_map, stpts_xyz, stpts_groups, _, scattered_features, stpts_xyz_folded = self.PG(corr_out['batch_p_2d_in_shape_order'].detach(), 
                                                                                                        shape, 
                                                                                                        corr_out['l_latent'].detach())
                
           
            self.num_unique_groups = self.PG.num_unique_groups
            
            stpts_prob_map_clamped = stpts_prob_map.clamp(1e-15, 1-1e-15)
            stpts_prob_map_tp = stpts_prob_map_clamped.transpose(1,2).contiguous()

            pg_entropy = Categorical(stpts_prob_map_clamped).entropy().mean()
            pg_cd = CD_loss(stpts_xyz_folded, shape)
            
            pg_loss = pg_cd # + 0.01 * pg_entropy

            self.log("train/pgloss", pg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train/pg_entropy_loss", pg_entropy, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train/pg_cd_loss", pg_cd, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train/groups", self.num_unique_groups, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
            if batch_idx % 20 == 0 and self.visualizer != None:
                self.visualizer.show_pointclouds(points=corr_out['self_rec_shape'][0], title="train/global_reconstruct")
                self.visualizer.show_sphere_groups(points=stpts_xyz_folded[0], title="train/stpts_xyz_folded", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="train/stpts_xyz", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=shape[0], title="train/grouped_shape", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=corr_out['unfold_pts'][0], title="train/grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_histogram(groups=stpts_groups[0], title="train/histogram", num_groups=num_groups)
                self.visualizer.show_heatmap(stpts_prob_map[0], title="train/heat")
                
            return pg_loss


        if optimizer_idx == 2: #fintune
            self.Encoder.eval()
            self.Unfold.eval()
            self.Fold.eval()
            self.PG.eval()
            # self.VQ.eval()
            # import pdb; pdb.set_trace

            # with torch.no_grad():
            corr_out = self.forward_correspondence(shape)
            
            # -------- PG -----------
            stpts_prob_map, stpts_xyz, stpts_groups, _, scattered_features, stpts_xyz_folded = self.PG(corr_out['batch_p_2d_in_shape_order'].detach(), 
                                                                                                        shape, 
                                                                                                        corr_out['l_latent'])
            # stpts_prob_map, stpts_xyz, stpts_groups, weighted_features, scattered_features, stpts_xyz_folded = self.PG(batch_p_2d_in_shape_order, shape, l_latent)
            # pg_cd = CD_loss(stpts_xyz_folded, shape)
        
            # -------- PG -----------
            _, _, _, grouped_features, _, _ = self.PG(corr_out['batch_p_2d'], shape, corr_out['l_latent_in_standard_order'])
            # _, _, _, _, scattered_features, _ = self.PG(batch_p_2d, shape, l_latent_in_standard_order)
            # scattered_features = corr_out['g_latent_stacked']

            # -------- VQ -----------
            # scattered_features_quant_conv = self.quant_conv(scattered_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            # grouped_features = self.quant_conv(grouped_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            # grouped_quant, emb_loss, info = self.VQ(grouped_features)
            grouped_quant, indices, emb_loss = self.VQ(grouped_features)
            # import pdb; pdb.set_trace()

            per_sample_codebook_usage = sum([indices[i].unique().size(0) for i in range(batch_size)])/batch_size
            per_batch_codebook_usage = indices.unique().size(0)

            # grouped_quant = self.post_quant_conv(grouped_quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

            quant = self.PG.remap_grouped_features(corr_out['batch_p_2d'], grouped_quant)
            # import pdb; pdb.set_trace()
            # quant = self.PG.remap_grouped_features(corr_out['batch_p_2d'], grouped_features)

            # quant = scattered_features_quant_conv
            # emb_loss = torch.tensor(0).to(quant)
            # Fold_f_input = self.post_quant_conv(quant.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

            # -------- Fold_f -----------
            # Fold_f_input =  batched_index_select(Fold_f_input, 1, standard_order.long()) # l_latent # scattered_features_quant_conv
            # Fold_f_input = corr_out['g_latent_stacked']
            # Fold_f_input = scattered_features
            Fold_f_input = quant
            self_rec_shape_f_ = self.Fold_f(Fold_f_input, corr_out['batch_p_2d']) 
            # import pdb; pdb.set_trace()
            # self_rec_shape_f_ = self.Fold_f(Fold_f_input, batch_p_2d_in_shape_order) 

            self_rec_shape_f_ = self.Fold_f(Fold_f_input, corr_out['batch_p_2d'])

            aeloss, log_dict_ae = self.ae_loss(emb_loss, 
                                                   corr_out['batch_p_2d'], 
                                                   shape, 
                                                   None, 
                                                   None, 
                                                   None, 
                                                   self_rec_shape_f_, 
                                                   optimizer_idx, self.global_step)

            # self_rec_shape_f_a = self.Fold_f(Fold_f_input, corr_out['batch_p_2d']*1.5)
            # self_rec_shape_f_b = self.Fold_f(Fold_f_input, corr_out['batch_p_2d']*0.5)
            # self_rec_shape_f_c = self.Fold_f(Fold_f_input, corr_out['batch_p_2d']*0.1)
            # self_rec_shape_f_d = self.Fold_f(Fold_f_input, corr_out['batch_p_2d']*4)
            # loss_sr_f_a = (CD_loss(self_rec_shape_f_a/1.5, shape) + EMD_loss(self_rec_shape_f_a/1.5, shape))
            # loss_sr_f_b = (CD_loss(self_rec_shape_f_b/0.5, shape) + EMD_loss(self_rec_shape_f_b/0.5, shape))
            # loss_sr_f_c = (CD_loss(self_rec_shape_f_c/0.1, shape) + EMD_loss(self_rec_shape_f_c/0.1, shape))
            # loss_sr_f_d = (CD_loss(self_rec_shape_f_d/4, shape) + EMD_loss(self_rec_shape_f_d/4, shape))*0.1
            aeloss = aeloss # + loss_sr_f_b + loss_sr_f_c # + loss_sr_f_d + loss_sr_f_a 
            # self.log("train/loss_sr_f_a", loss_sr_f_a, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # self.log("train/loss_sr_f_b", loss_sr_f_b, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # self.log("train/loss_sr_f_c", loss_sr_f_c, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            # self.log("train/loss_sr_f_d", loss_sr_f_d, prog_bar=False, logger=True, on_step=False, on_epoch=True)

            self.log("train/aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True) 
            self.log("train/per_sample_codebook_usage", per_sample_codebook_usage, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log("train/per_batch_codebook_usage", per_batch_codebook_usage, prog_bar=False, logger=True, on_step=False, on_epoch=True)  
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # self.log("train/pg_cd_san", pg_cd, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            if batch_idx % 20 == 0 and self.visualizer != None:
                self.visualizer.show_pointclouds(points=shape[0], title="train/input")
                self.visualizer.show_pointclouds(points=corr_out['self_rec_shape'][0], title="train/global_reconstruct")
                # self.visualizer.show_pointclouds(points=self_rec_shape_u[0], title="train/global_reconstruct_u")
                # self.visualizer.show_pointclouds(points=unfold_pts[0], title="train/global_unfold")
                # self.visualizer.show_pointclouds(points=batch_p_2d[0], title="train/unfold_target")
                self.visualizer.show_pointclouds(points=self_rec_shape_f_[0], title="train/self_rec_shape_f_")

                self.visualizer.show_sphere_groups(points=stpts_xyz_folded[0], title="train/stpts_xyz_folded", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=stpts_xyz[0], title="train/stpts_xyz", groups=torch.tensor(np.arange(num_groups)), num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=shape[0], title="train/grouped_shape", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_sphere_groups(points=corr_out['unfold_pts'][0], title="train/grouped_sphere", groups=stpts_groups[0], num_groups=num_groups)
                self.visualizer.show_histogram(groups=stpts_groups[0], title="train/histogram", num_groups=num_groups)
            
            return aeloss



        if optimizer_idx == 3:
            self.Encoder.eval()
            self.Fold.eval()
            # self.Unfold.eval()
            self.PG.eval()
            self.VQ.eval()
            # self.quant_conv.eval()
            # self.post_quant_conv.eval()

            if self.condition == 'mask':
                image = batch.get('image').float().view(-1, 3, 480, 480)
                self.cond_feature_extractor.eval()
                with torch.no_grad():
                    cond_feature = self.cond_feature_extractor(image).flatten(1)
                cond_feature = self.cond_feature_projection(cond_feature)
            else:
                cond_feature = None

            with torch.no_grad():
                corr_out = self.forward_correspondence(shape)
                
                _, _, _, grouped_features, _, _ = self.PG(corr_out['batch_p_2d'], shape, corr_out['l_latent_in_standard_order'])
                grouped_quant, indices, _ = self.VQ(grouped_features, pad_index=False)

                reordered_indices = indices[:, self.PG.fibonacci_grid_groups_unique.long()] # .permute(0,2,1)
                z_indices = reordered_indices.view(shape.size(0),-1) # [B, num, _unq_g] 
            
            _, c_indices = self.encode_c(shape) # c_indices = [B, 1]

            self.pkeep = 1.0
            if self.training and self.pkeep < 1.0:
                mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape, device=z_indices.device))
                mask = mask.round().to(dtype=torch.int64)
                r_indices = torch.randint_like(z_indices, self.Transformer.config.vocab_size)
                a_indices = mask*z_indices+(1-mask)*r_indices
            else:
                a_indices = z_indices

            cz_indices = torch.cat((c_indices, a_indices), dim=1)  
            
            # make the prediction
            logits, _ = self.Transformer(cz_indices[:, :-1], num_unique_groups=self.PG.num_unique_groups, embeddings=cond_feature)

            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c) 
            logits = logits[:, c_indices.shape[1]-1:]
            # import pdb; pdb.set_trace()
            # transformer prediction loss
            tf_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))

            self.log("train/tfloss", tf_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # print(z_indices)

            return tf_loss

    @torch.no_grad()
    def encode_c(self, inputs):
        ''' Encodes the input to condition c.
            If unconditional, return SOS token instead
        '''
        quant_c, _, [_,_,indices] = self.cond_model.encode(inputs)
        if len(indices.shape) > 2:
            indices = indices.view(inputs.shape[0], -1)
        return quant_c, indices



    def test_step(self, batch, batch_idx):
        rt = self(batch, denorm=True)

        if batch_idx % 20 == 0 and self.visualizer != None:
            self.visualizer.show_sphere_groups(points=rt['info']['stpts_xyz_folded'][0], title="test/stpts_xyz_folded", groups=torch.tensor(np.arange(self.num_groups)), num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['info']['stpts_xyz'][0], title="test/stpts_xyz", groups=torch.tensor(np.arange(self.num_groups)), num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['global_rec_shape'][0], title="test/grouped_shape", groups=rt['info']['stpts_groups'][0], num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['info']['unfold_pts'][0], title="test/grouped_sphere", groups=rt['info']['stpts_groups'][0], num_groups=self.num_groups)
            self.visualizer.show_histogram(groups=rt['info']['stpts_groups'][0], title="test/histogram", num_groups=self.num_groups)
            
            if self.mode == 3:
                if self.condition == None:
                    for b in range(10):
                        gen_shape = rt['gen_our_shape'][b]
                        stpts_groups = rt['gen']['stpts_groups'][b]
                        self.visualizer.show_sphere_groups(points=gen_shape, title="eval_gen_pts_{:02d}".format(b), groups=stpts_groups, num_groups=self.num_groups)
                elif self.condition == 'mask':
                    for b in range(3):
                        for v in range(5):
                            stpts_groups = rt['gen']['stpts_groups'][b]
                            self.visualizer.vis.image(rt['gen']['completion_image'][b][v], win="eval_img_{:02d}_{:02d}".format(b, v), opts=dict(title="eval_img_{:02d}_{:02d}".format(b, v)))
                            self.visualizer.show_sphere_groups(points=torch.tensor(rt['completion_gen_shape'][b][v]), title="eval_gen_pts_{:02d}_{:02d}".format(b,v), groups=stpts_groups, num_groups=self.num_groups)

        return rt

    def test_epoch_end(self, validation_step_outputs):
        all_rec = list()
        all_ref = list()
        others = {}
        vq_indices = list()
        img_list = list()

        for step in range(len(validation_step_outputs)):
            validation_step_output = validation_step_outputs[step]
            for k in validation_step_output.keys():
                if k == "gdt_shape":
                    all_ref.append(validation_step_output[k])
                elif k == "rec_shape":
                    all_rec.append(validation_step_output[k])
                elif "shape" in k:
                    if k in others:
                        others[k].append(validation_step_output[k])
                    else:
                        others[k] = list()
                        others[k].append(validation_step_output[k])

                if k == "info":
                    vq_indices.append(validation_step_output[k]['vq_indices'])

                if k == "gen":
                    if self.condition == 'mask':
                        img_list.append(validation_step_output[k]['completion_image'])

        # per_set_codebook_usage = torch.concat(vq_indices).flatten().unique().size(0)
        per_set_codebook_usage = []
        codebook_list = torch.concat(vq_indices)
        for g_id in range(self.num_groups):
            per_set_codebook_usage.append(codebook_list[:, g_id].unique().size(0))

        self.log("test/per_set_codebook_usage", np.mean(per_set_codebook_usage), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        results = {}
        all_ref = torch.cat(all_ref, 0)
        all_rec = torch.cat(all_rec, 0)
        # print("Reconstruction Evaluation start")
        # print(f"Reconstruction Sample size:{all_rec.size()} Ref size: {all_ref.size()}")

        emd_cd_rt = emd_cd(all_rec, all_ref, 128, accelerated_cd=True)
        emd_cd_rt = {
            'test/'+k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in emd_cd_rt.items()}
        
        self.log("test/CD", emd_cd_rt["test/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/EMD", emd_cd_rt["test/EMD"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        print(emd_cd_rt)
        # results.update(emd_cd_rt)

        for k in others.keys():
            if "gen" in k or "completion" in k:
                continue
            all_other = torch.cat(others[k], 0)
            # print("Reconstruction Evaluation (", k,") start")
            # print(f"Reconstruction Other. Sample size:{all_other.size()} Ref size: {all_ref.size()}")
            emd_cd_rt = emd_cd(all_other, all_ref, 128, accelerated_cd=True)
            emd_cd_rt = {
                'test/'+k+'_'+j: (v.cpu().detach().item() if not isinstance(v, float) else v)
                for j, v in emd_cd_rt.items()}
            
            results.update(emd_cd_rt)

        np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_rec.npy'), all_rec.cpu().numpy())
        np.save(os.path.join(self.logger.save_dir, 'gdt_origin_scale.npy'), all_ref.cpu().numpy())
        np.save(os.path.join(self.logger.save_dir, 'per_set_codebook_usage.npy'), per_set_codebook_usage)
        # np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_rec.npy'), all_rec.cpu().numpy())
        # self.log("val/CD", results["val/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/EMD", results["val/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.mode == 3:
            # import pdb; pdb.set_trace()
            if self.condition == 'mask':
                # all_completion_ref_shape = torch.tensor(np.concatenate(others["completion_ref_shape"])).cuda().view(-1, 2048, 3)
                # all_completion_gen_shape = torch.tensor(np.concatenate(others["completion_gen_shape"])).cuda()
                all_completion_ref_shape = np.concatenate(others["completion_ref_shape"])
                all_completion_gen_shape = np.concatenate(others["completion_gen_shape"])
                all_completion_gen_shape_norm = np.concatenate(others["completion_gen_shape_norm"])

                # emd_cd_rt = emd_cd(all_completion_gen_shape, all_completion_ref_shape, 128, accelerated_cd=True)
                # emd_cd_rt = {
                #     'test/completion_'+k+'_'+j: (v.cpu().detach().item() if not isinstance(v, float) else v)
                #     for j, v in emd_cd_rt.items()}
                
                # results.update(emd_cd_rt)

                # import pdb; pdb.set_trace()
                # all_images = np.concatenate(img_list[:100])
                np.save(os.path.join(self.logger.save_dir, 'all_completion_ref_shape.npy'), all_completion_ref_shape)
                np.save(os.path.join(self.logger.save_dir, 'all_completion_gen_shape.npy'), all_completion_gen_shape)
                np.save(os.path.join(self.logger.save_dir, 'all_completion_gen_shape_norm.npy'), all_completion_gen_shape_norm)

            elif self.condition == None:
                
                all_ref_normalized = torch.cat(others["gen_ref_shape"], 0)
                all_gen_normalized = torch.cat(others["gen_our_shape"], 0)

                np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_gen.npy'), all_gen_normalized.cpu().numpy())
                np.save(os.path.join(self.logger.save_dir, 'gdt_normed_scale.npy'), all_ref_normalized.cpu().numpy())

                # print("Generation Evaluation start")
                # print(f"Generation Sample size:{all_gen_normalized.size()} Ref size: {all_ref_normalized.size()}")
                gen_results = compute_all_metrics(all_gen_normalized, all_ref_normalized, 256, accelerated_cd=True, compute_nna=True, compute_jsd=True)
                gen_results = {'test/'+k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in gen_results.items()}
                print(gen_results)
                # jsd = JSD(all_gen_normalized.cpu().detach().numpy(), all_ref_normalized.cpu().detach().numpy())
                # gen_results.update({'test/JSD': jsd})
                results.update(gen_results)

                # import pdb; pdb.set_trace()

        self.log_dict(results, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return self.log_dict





    def validation_step(self, batch, batch_idx):
        rt = self(batch, denorm=True)

        if batch_idx % 20 == 0 and self.visualizer != None:
            self.visualizer.show_sphere_groups(points=rt['info']['stpts_xyz_folded'][0], title="val/stpts_xyz_folded", groups=torch.tensor(np.arange(self.num_groups)), num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['info']['stpts_xyz'][0], title="val/stpts_xyz", groups=torch.tensor(np.arange(self.num_groups)), num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['oracle_shape'][0], title="val/grouped_shape", groups=rt['info']['stpts_groups'][0], num_groups=self.num_groups)
            self.visualizer.show_sphere_groups(points=rt['info']['unfold_pts'][0], title="val/grouped_sphere", groups=rt['info']['stpts_groups'][0], num_groups=self.num_groups)
            self.visualizer.show_histogram(groups=rt['info']['stpts_groups'][0], title="val/histogram", num_groups=self.num_groups)
            
            self.visualizer.show_pointclouds(points=rt['global_rec_shape'][0], title="val/global_rec")
            self.visualizer.show_pointclouds(points=rt['rec_shape'][0], title="val/local_rec")

            if self.mode == 3:
                if self.condition == None:
                    for b in range(10):
                        gen_shape = rt['gen_our_shape'][b]
                        stpts_groups = rt['gen']['stpts_groups'][b]
                        self.visualizer.show_sphere_groups(points=gen_shape, title="eval_gen_pts_{:02d}".format(b), groups=stpts_groups, num_groups=self.num_groups)
                elif self.condition == 'mask':
                    for b in range(10):
                        for v in range(1):
                            stpts_groups = rt['gen']['stpts_groups'][b]
                            self.visualizer.vis.image(rt['gen']['completion_image'][b][v], win="eval_img_{:02d}_{:02d}".format(b, v), opts=dict(title="eval_img_{:02d}_{:02d}".format(b, v)))
                            self.visualizer.show_sphere_groups(points=torch.tensor(rt['completion_gen_shape'][b][v]), title="eval_gen_pts_{:02d}_{:02d}".format(b,v), groups=stpts_groups, num_groups=self.num_groups)

        return rt

    def validation_epoch_end(self, validation_step_outputs):
        all_rec = list()
        all_ref = list()
        others = {}
        vq_indices = list()
        img_list = list()

        for step in range(len(validation_step_outputs)):
            validation_step_output = validation_step_outputs[step]
            for k in validation_step_output.keys():
                if k == "gdt_shape":
                    all_ref.append(validation_step_output[k])
                elif k == "rec_shape":
                    all_rec.append(validation_step_output[k])
                elif "shape" in k:
                    if k in others:
                        others[k].append(validation_step_output[k])
                    else:
                        others[k] = list()
                        others[k].append(validation_step_output[k])
                
                if k == "info":
                    vq_indices.append(validation_step_output[k]['vq_indices'])

                if k == "gen":
                    if self.condition == 'mask':
                        img_list.append(validation_step_output[k]['completion_image'])

        # per_set_codebook_usage = torch.concat(vq_indices).flatten().unique()
        # self.log("val/per_set_codebook_usage", per_set_codebook_usage, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        per_set_codebook_usage = []
        codebook_list = torch.concat(vq_indices)
        for g_id in range(self.num_groups):
            per_set_codebook_usage.append(codebook_list[:, g_id].unique().size(0))

        self.log("val/per_set_codebook_usage", np.mean(per_set_codebook_usage), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        results = {}
        all_ref = torch.cat(all_ref, 0)
        all_rec = torch.cat(all_rec, 0)
        # print("Reconstruction Evaluation start")
        # print(f"Reconstruction Sample size:{all_rec.size()} Ref size: {all_ref.size()}")

        emd_cd_rt = emd_cd(all_rec, all_ref, 128, accelerated_cd=True)
        emd_cd_rt = {
            'val/'+k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in emd_cd_rt.items()}
        
        self.log("val/CD", emd_cd_rt["val/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/EMD", emd_cd_rt["val/EMD"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        # results.update(emd_cd_rt)

        for k in others.keys():
            if "gen" in k or "completion" in k:
                continue
            all_other = torch.cat(others[k], 0)
            # print("Reconstruction Evaluation (", k,") start")
            # print(f"Reconstruction Other. Sample size:{all_other.size()} Ref size: {all_ref.size()}")
            emd_cd_rt = emd_cd(all_other, all_ref, 128, accelerated_cd=True)
            emd_cd_rt = {
                'val/'+k+'_'+j: (v.cpu().detach().item() if not isinstance(v, float) else v)
                for j, v in emd_cd_rt.items()}
            
            results.update(emd_cd_rt)
            # import pdb; pdb.set_trace()
        # np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_rec.npy'), all_rec.cpu().numpy())
        # self.log("val/CD", results["val/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/EMD", results["val/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.mode == 3:

            if self.condition == 'mask':
                all_completion_ref_shape = torch.tensor(np.concatenate(others["completion_ref_shape"])).cuda().view(-1, 2048, 3).contiguous()
                all_completion_gen_shape = torch.tensor(np.concatenate(others["completion_gen_shape"])).cuda()[:,:,1,:,:].view(-1, 2048, 3).contiguous()

                emd_cd_rt = emd_cd(all_completion_gen_shape, all_completion_ref_shape, 128, accelerated_cd=True)
                emd_cd_rt = {
                    'val/C_'+k: (v.cpu().detach().item() if not isinstance(v, float) else v)
                    for k, v in emd_cd_rt.items()}
                
                self.log("val/C_CD", emd_cd_rt["val/C_CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val/C_EMD", emd_cd_rt["val/C_EMD"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

            elif self.condition == None:

                all_ref_normalized = torch.cat(others["gen_ref_shape"], 0)
                all_gen_normalized = torch.cat(others["gen_our_shape"], 0)
                # print("Generation Evaluation start")
                # print(f"Generation Sample size:{all_gen_normalized.size()} Ref size: {all_ref_normalized.size()}")
                gen_results = compute_all_metrics(all_gen_normalized, all_ref_normalized, 256, accelerated_cd=True)
                gen_results = {'val/'+k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in gen_results.items()}
                # print(gen_results)
                results.update(gen_results)
                # np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_gen.npy'), all_gen_normalized.cpu().numpy())
                # np.save(os.path.join(self.logger.save_dir, str(self.current_epoch)+'_rec.npy'), all_rec.cpu().numpy())

        self.log_dict(results, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return self.log_dict


    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
            exit()

    def configure_optimizers(self):
        lr = self.learning_rate / 4
        if self.no_fold:
            opt_corr = torch.optim.Adam( # list(self.Fold_f.parameters())+
                                        list(self.Encoder.parameters())+
                                        list(self.Unfold.parameters()),
                                        lr=0.001)
            corr_scheduler = optim.lr_scheduler.StepLR(opt_corr, step_size=200, gamma=0.1)
        else:
            opt_corr = torch.optim.Adam( # list(self.Fold_f.parameters())+
                                        list(self.Encoder.parameters())+
                                        list(self.Fold.parameters()),
                                        # list(self.Unfold.parameters()),
                                    #   list(self.VQ.parameters())+
                                    #   list(self.quant_conv.parameters())+
                                    #   list(self.post_quant_conv.parameters()),
                                        lr=5e-4) # 0.001
            corr_scheduler = optim.lr_scheduler.StepLR(opt_corr, step_size=200, gamma=0.1)

        opt_ft = torch.optim.Adam(list(self.Fold_f.parameters())+
                                list(self.Encoder_f.parameters())+
                                list(self.VQ.parameters()),
                                #   list(self.quant_conv.parameters()),
                                #   list(self.post_quant_conv.parameters()),
                                lr=5e-4) # 5e-4 4.5e-6
        
        ft_scheduler = optim.lr_scheduler.StepLR(opt_ft, step_size=1500, gamma=0.1) # 1500

        if self.mode == 1:
            opt_pg = torch.optim.Adam(
                self.PG.parameters(), lr=5e-4   # 5e-4
            )
            pg_sched = optim.lr_scheduler.StepLR(opt_pg, step_size=100, gamma=0.5)

        if self.mode == 0:
            return [opt_corr], [corr_scheduler]
        if self.mode == 1:
            return [opt_pg], [pg_sched]
        if self.mode == 2:
            return [opt_ft], [ft_scheduler]
        # if self.mode == 1:
        #     return [opt_ae, opt_pg], []
        if self.mode == 3:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.Transformer.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # if self.condition == 'mask':
            #     for mn, m in self.cond_feature_projection.named_modules():
            #         for pn, p in m.named_parameters():
            #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            #             if pn.endswith('bias'):
            #                 # all biases will not be decayed
            #                 no_decay.add(fpn)
            #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
            #                 # weights of whitelist modules will be weight decayed
            #                 decay.add(fpn)
            #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
            #                 # weights of blacklist modules will NOT be weight decayed
            #                 no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('pos_emb')

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.Transformer.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object

            if self.condition == 'mask':
                optim_groups = [
                    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
                    {"params": self.cond_feature_projection.parameters(), "weight_decay": 0.01},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
                ]
            opt_tf = torch.optim.AdamW(optim_groups, lr=4.5e-5 , betas=(0.9, 0.95)) # 4.5e-6 # 5e-4 !!! start from --> 4.5e-4 !!!
            print("using opt_tf")
            tf_scheduler = optim.lr_scheduler.StepLR(opt_tf, step_size=1000, gamma=0.1)
            return [opt_tf], [tf_scheduler]
        # else:
        #     return [opt_ae], []
        # return [opt_ae, opt_pg], []

class LRPolicy(object):
    def __init__(self, rate=0.7):
        self.rate = rate

    def __call__(self, it):
        rt = max(
            self.rate ** (int(it * 16 / 500)),
            1e-5 / 1e-2,
        )
        return rt

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps=0, lr_min=1.0e-06, lr_max=0.9, lr_start=0.9, max_decay_steps=5000, verbosity_interval=10000):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)