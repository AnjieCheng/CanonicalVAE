
import torch
import torch.nn.functional as F
import external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from external.metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD, emd_cd
from external.emd.emd_module import *
from src.utils.common import *

criterion = torch.nn.MSELoss()

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def cosine_loss(N1, N2):
    loss = N1*N2
    loss = loss.sum(-1)
    loss = torch.abs(loss)
    loss = 1-loss
    loss = torch.mean(1-F.cosine_similarity(N1, N2))
    return torch.mean(loss)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    '''
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def farthest_point_sample(xyz, npoint):
    '''
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    centroids_feat = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


#------------------------------------------------------------------------------------------------------------#

def unfold_loss(esti_shapes, shapes, full_loss=False):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes) # idx1[16, 2048] idx2[16, 2562]
    if full_loss:
        loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    else:
        loss_cd = torch.mean(torch.sqrt(dist1))
    return loss_cd

def occupancy_loss(esti_values, values):
    return criterion(esti_values, values)

def selfrec_loss(esti_shapes, shapes):
    return criterion(esti_shapes, shapes)

def CD_loss(esti_shapes, shapes):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    return loss_cd

def CD_loss_min(esti_shapes, shapes):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    loss_cd = torch.min(torch.mean(torch.sqrt(dist1)), torch.mean(torch.sqrt(dist2)))
    return loss_cd

def CD_normal_loss(esti_shapes, shapes, esti_normals, normals):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))

    corr_normal = torch.gather(normals, 1, idx1.long().unsqueeze(-1).repeat(1,1,3))
    loss_normal = cosine_loss(esti_normals, corr_normal)
    return loss_cd, loss_normal

def get_l_latent_by_chamfer_index(unfold_pts, batch_p_2d, l_latent):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(unfold_pts, batch_p_2d)
    l_latent_query = batched_index_select(l_latent, 1, idx2.long())
    return l_latent_query, idx1, idx2

def get_order_by_chamfer_index(unfold_pts, batch_p_2d):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(unfold_pts, batch_p_2d)
    return idx1, idx2

def EMD_loss(esti_shapes, shapes):
    emd_dist = emdModule()
    dist, assigment = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()
    return loss_emd

class CanonicalVQLossAll(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_codebook, batch_p_2d, shape, self_rec_shape, unfold_pts, self_rec_shape_u, self_rec_shape_f_, optimizer_idx, global_step, split="train"):
        loss_sr = CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape) # V
        loss_unfold = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d) # V
        loss_sr_u = selfrec_loss(self_rec_shape_u, shape) +  CD_loss(self_rec_shape_u, shape) + EMD_loss(self_rec_shape_u, shape)
        loss_sr_f_ = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape) 
        loss = loss_codebook + loss_sr + loss_unfold + loss_sr_u + loss_sr_f_ 

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): loss_codebook.detach().mean(),
                "{}/loss_sr".format(split): loss_sr.detach().mean(),
                "{}/loss_unfold".format(split): loss_unfold.detach().mean(),
                "{}/loss_sr_u".format(split): loss_sr_u.detach().mean(),
                "{}/loss_sr_f_".format(split): loss_sr_f_.detach().mean(),
                }
        return loss, log


class CanonicalVQLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_codebook, batch_p_2d, shape, self_rec_shape, unfold_pts, self_rec_shape_u, self_rec_shape_f_, optimizer_idx, global_step, split="train"):
        loss_sr_f_ = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape)
        loss = loss_sr_f_ + loss_codebook 

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): loss_codebook.detach().mean(),
                "{}/loss_sr_f_".format(split): loss_sr_f_.detach().mean(),
                }
        return loss, log

class CorrespondenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, shape, batch_p_2d, self_rec_shape, unfold_pts, self_rec_shape_u, split="train", train_unfold=True, no_fold=False):
        if no_fold:
            loss_unfold = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d) # V
            loss = loss_unfold
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/loss_unfold".format(split): loss_unfold.detach().mean(),
                    }
        else:
            loss_sr = CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape) # V
            if train_unfold:
                loss_unfold = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d) # V
                loss_sr_u = selfrec_loss(self_rec_shape_u, shape) +  CD_loss(self_rec_shape_u, shape) + EMD_loss(self_rec_shape_u, shape)
                loss = loss_sr + loss_unfold + loss_sr_u 

                log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                        "{}/loss_sr".format(split): loss_sr.detach().mean(),
                        "{}/loss_unfold".format(split): loss_unfold.detach().mean(),
                        "{}/loss_sr_u".format(split): loss_sr_u.detach().mean(),
                        }

            else:
                loss = loss_sr
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                        "{}/loss_sr".format(split): loss_sr.detach().mean(),
                        }
        return loss, log
