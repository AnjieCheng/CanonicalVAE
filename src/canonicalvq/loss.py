
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
        # loss_pg = CD_loss(stpts_xyz_folded, shape)
        loss_sr_f_ = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape) # + selfrec_loss(self_rec_shape_f_, shape)
        loss = loss_codebook + loss_sr + loss_unfold + loss_sr_u + loss_sr_f_ # + loss_pg

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): loss_codebook.detach().mean(),
                "{}/loss_sr".format(split): loss_sr.detach().mean(),
                "{}/loss_unfold".format(split): loss_unfold.detach().mean(),
                "{}/loss_sr_u".format(split): loss_sr_u.detach().mean(),
                # "{}/loss_pg".format(split): loss_pg.detach().mean(),
                "{}/loss_sr_f_".format(split): loss_sr_f_.detach().mean(),
                }
        return loss, log


class CanonicalVQLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_codebook, batch_p_2d, shape, self_rec_shape, unfold_pts, self_rec_shape_u, self_rec_shape_f_, optimizer_idx, global_step, split="train"):
        # loss_sr = CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape) # V
        # loss_unfold = CD_loss(unfold_pts, batch_p_2d) + EMD_loss(unfold_pts, batch_p_2d) # V
        # loss_sr_u = selfrec_loss(self_rec_shape_u, shape) +  CD_loss(self_rec_shape_u, shape) + EMD_loss(self_rec_shape_u, shape)
        loss_sr_f_ = CD_loss(self_rec_shape_f_, shape) + EMD_loss(self_rec_shape_f_, shape) # + selfrec_loss(self_rec_shape_f_, shape) # + selfrec_loss(self_rec_shape_f_, shape)
        loss = loss_sr_f_ + loss_codebook # + loss_sr + loss_unfold + loss_sr_u # + loss_pg

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): loss_codebook.detach().mean(),
                # "{}/loss_sr".format(split): loss_sr.detach().mean(),
                # "{}/loss_unfold".format(split): loss_unfold.detach().mean(),
                # "{}/loss_sr_u".format(split): loss_sr_u.detach().mean(),
                "{}/loss_sr_f_".format(split): loss_sr_f_.detach().mean(),
                }
        return loss, log

class CorrespondenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, shape, batch_p_2d, self_rec_shape, unfold_pts, self_rec_shape_u, split="train", train_unfold=True, no_fold=False):
        # ec = emd_cd(self_rec_shape, shape, batch_size=16)
        # loss_sr = ec['CD'] + ec['EMD']
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
                loss = loss_sr + loss_unfold + loss_sr_u # + loss_pg

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


from torch.autograd import Variable
def gen_loss(d_real, d_fake, gan="wgan", weight=1., d_real_p=None, d_fake_p=None,noise_label=False):
    if gan.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif gan.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "ls":
        #mse = nn.MSELoss()
        B = d_fake.size(0)
        #real_label_np = np.ones((B,))
        fake_label_np = np.ones((B,))

        if noise_label:
            # occasionally flip the labels when training the generator to fool the D
            fake_label_np = noisy_labels(fake_label_np, 0.05)

        #real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()

        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())
        g_loss = F.mse_loss(d_fake, fake_label)



        if d_fake_p is not None:
            fake_label_p = Variable(torch.FloatTensor(d_fake_p.size(0), d_fake_p.size(1)).fill_(1).cuda())
            g_loss_p = F.mse_loss(d_fake_p,fake_label_p)
            g_loss = g_loss + 0.2*g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "gan":
        fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target=fake_target)
        g_loss = fake_loss(d_fake)

        if d_fake_p is not None:
            g_loss_p = fake_loss(d_fake_p.view(-1))
            g_loss = g_loss + g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        # https://github.com/weishenho/SAGAN-with-relativistic/blob/master/main.py
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss =  torch.mean((d_real - torch.mean(d_fake) + y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) - y) ** 2)

        # d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        # g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss = (g_loss + d_loss) / 2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)

def dis_loss(d_real, d_fake, gan="wgan", weight=1.,d_real_p=None, d_fake_p=None, noise_label=False):
    # B = d_fake.size(0)
    # a = 1.0
    # b = 0.9

    if gan.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif gan.lower() == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()

        # d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        # d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        real_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        real_acc = real_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": real_acc.clone().detach(),
            "dis_correct": real_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    elif gan.lower() == "ls":
        mse = nn.MSELoss()
        B = d_fake.size(0)

        real_label_np = np.ones((B,))
        fake_label_np = np.zeros((B,))

        if noise_label:
            real_label_np = smooth_labels(B,ran=[0.9,1.0])
            #fake_label_np = smooth_labels(B,ran=[0.0,0.1])
            # occasionally flip the labels when training the D to
            # prevent D from becoming too strong
            real_label_np = noisy_labels(real_label_np, 0.05)
            #fake_label_np = noisy_labels(fake_label_np, 0.05)


        real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()
        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()


        # real_label = Variable((1.0 - 0.9) * torch.rand(d_fake.size(0)) + 0.9).cuda()
        # fake_label = Variable((0.1 - 0.0) * torch.rand(d_fake.size(0)) + 0.0).cuda()

        t = 0.5
        real_correct = (d_real >= t).float().sum()
        real_acc = real_correct / float(d_real.size(0))

        fake_correct  = (d_fake < t).float().sum()
        fake_acc = fake_correct / float(d_fake.size(0))
        # + d_fake.size(0))

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())

        g_loss = F.mse_loss(d_fake, fake_label)
        d_loss = F.mse_loss(d_real, real_label)

        if d_real_p is not None and d_fake_p is not None:

            real_label_p = Variable((1.0 - 0.9) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.9).cuda()
            fake_label_p = Variable((0.1 - 0.0) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.0).cuda()

            # real_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(1).cuda())
            # fake_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(0).cuda())
            g_loss_p = F.mse_loss(d_fake_p, fake_label_p)
            d_loss_p = F.mse_loss(d_real_p, real_label_p)

            g_loss = (g_loss + 0.1*g_loss_p)
            d_loss = (d_loss + 0.1*d_loss_p)

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach(),
            "fake_acc": fake_acc.clone().detach(),
            "real_acc": real_acc.clone().detach()
        }
    elif gan.lower() =="gan":
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)

        g_loss, d_loss = discriminator_loss(d_fake, d_real)

        if d_real_p is not None and d_fake_p is not None:
            g_loss_p,d_loss_p = discriminator_loss(d_fake_p.view(-1),d_real_p.view(-1))
            g_loss = (g_loss + g_loss_p)/2.0
            d_loss = (d_loss + d_loss_p)/2.0

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss =  (g_loss+d_loss)/2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)
