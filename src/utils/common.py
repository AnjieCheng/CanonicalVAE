import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import scipy
import os
from datetime import datetime
import shutil

# from pointnet2_ops import pointnet2_utils

def batch_sample_from_grid(grid, batch_size, K=None, downsample=False, grid_dim=3):
    grid_size = grid.shape[0]

    grid = grid.unsqueeze(0)
    grid = grid.expand(batch_size, -1, -1) # BxNx2

    if not downsample:
        assert K == None
        return grid
        
    assert(grid_size >= K)
    idx = torch.randint(
        low=0, high=grid_size,
        size=(batch_size, K),
    )

    idx, _ = torch.sort(idx, 1)
    idx = idx[:, :, None].expand(batch_size, K, grid_dim).to(grid.device)
    sampled_points = torch.gather(grid, dim=1, index=idx)
    
    assert(sampled_points.size() == (batch_size, K, grid_dim))
    return sampled_points

def denormalize(input_shape, loc, scale):
    return (input_shape * scale) + loc

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c

def match_source_to_target_points(source, target, device):
    indices_batch, _ = get_nearest_neighbors_indices_batch(source.cpu().numpy(), target.cpu().numpy())
    indices_batch_ts = torch.tensor(np.stack(indices_batch,axis=0).astype(np.int32)).long().to(device)
    matched_batch = batched_index_select(target, 1, indices_batch_ts)
    return indices_batch_ts, matched_batch

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.
    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        # kdtree = KDTree(p2)
        dist, idx = scipy.spatial.KDTree(p2).query(p1)      # kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


# def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
#     '''
#     :param pts: bn, n, 3
#     :param num_samples:
#     :return:
#     '''
#     sampled_pts_idx = pointnet2_utils.furthest_point_sample(pts, num_samples) # furthest_point_sample
#     sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
#     batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
#     batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
#     sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
#     sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

#     if return_sampled_idx == False:
#         return sampled_pts
#     else:
#         return sampled_pts, sampled_pts_idx

def backup_code(out_dir, basefile):
    code_dir = os.path.join(out_dir, 'code')
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)

    backup_dir = os.path.join(
        code_dir,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(backup_dir)

    shutil.copy(os.path.basename(basefile), backup_dir)
    shutil.copytree('./src', os.path.join(backup_dir, 'src')) 