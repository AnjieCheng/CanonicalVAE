"""
Uniform15KPC dataset
References: https://github.com/stevenygd/PointFlow, https://github.com/jw9730/setvae
"""
import os, logging
import random
import numpy as np
from src.data import point_operation

import torch
from torch.utils import data

logger = logging.getLogger(__name__)


# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    'all': 'all'
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

class Uniform15KPC(data.Dataset):
    ''' Uniform15KPC dataset class.
    '''

    def __init__(self, root, subdirs, tr_sample_size=10000, te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None, input_dim=3, standardize_per_shape=False, no_normalize=False):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        if split == 'train':
            self.max = tr_sample_size
        elif split == 'val':
            self.max = te_sample_size
        else:
            self.max = max((tr_sample_size, te_sample_size))

        sample_count = 0
        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                obj_fname = os.path.join(root, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))
                sample_count += 1

        # Shuffle the index deterministically (based on the number of examples)
        # self.all_points = self.all_points[:50]

        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)

        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d" % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale    

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def save_statistics(self, save_dir):
        np.save(os.path.join(save_dir, f"{self.split}_set_mean.npy"), self.all_points_mean)
        np.save(os.path.join(save_dir, f"{self.split}_set_std.npy"), self.all_points_std)
        np.save(os.path.join(save_dir, f"{self.split}_set_idx.npy"), np.array(self.shuffle_idx))

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        # ### test upper-bound ###
        # tr_out_random = self.train_points[idx]
        # tr_idxs_random = np.random.choice(tr_out_random.shape[0], self.tr_sample_size)
        # tr_out_random = torch.from_numpy(tr_out_random[tr_idxs_random, :]).float()

        tr_ofs = tr_out.mean(0, keepdim=True)
        te_ofs = te_out.mean(0, keepdim=True)

        # m, s = self.get_pc_stats(idx)
        # m, s = torch.from_numpy(np.asarray(m)), torch.from_numpy(np.asarray(s))

        shift, scale = self.get_standardize_stats(idx)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))
        
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'pointcloud': tr_out, # if self.split == 'train' else te_out
            'pointcloud_ref': te_out,
            'offset': tr_ofs if self.split == 'train' else te_ofs,
            'label': cate_idx,
            'sid': sid, 'mid': mid,
            'shift': shift, 'scale': scale,
            # 'pointcloud_random': tr_out_random,
        }

class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root="/work/vslab2018/3d/data/ShapeNetCore.v2.PC15k/",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None, standardize_per_shape=True, no_normalize=True):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        assert 'v2' in root, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super().__init__(
            root, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3,
            standardize_per_shape=standardize_per_shape, 
            no_normalize=no_normalize)


def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    shift = torch.stack(ret['shift'], dim=0)  # [B, 1, 3]
    scale = torch.stack(ret['scale'], dim=0)  # [B, 1, 1]

    s = torch.stack(ret['set'], dim=0)  # [B, N, 3]
    offset = torch.stack(ret['offset'], dim=0)
    mask = torch.zeros(s.size(0), s.size(1)).bool()  # [B, N]
    cardinality = torch.ones(s.size(0)) * s.size(1)  # [B,]

    ret.update({'pointcloud': s, 'offset': offset, 'set_mask': mask, 'cardinality': cardinality,
                'shift': shift, 'scale': scale})
    return ret


def build(cfg):
    val_dataset = ShapeNet15kPointClouds(
        categories=cfg['data']['cates'],
        split='val',
        tr_sample_size=cfg['data']['tr_max_sample_points'],
        te_sample_size=cfg['data']['te_max_sample_points'],
        scale=cfg['data']['dataset_scale'],
        root=cfg['data']['shapenet_data_dir'],
        normalize_per_shape=cfg['data']['normalize_per_shape'],
        normalize_std_per_axis=cfg['data']['normalize_std_per_axis'],
        standardize_per_shape=cfg['data']['standardize_per_shape'],
        no_normalize=cfg['data']['no_normalize'])

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    vis_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    train_dataset = train_loader = None
    return train_dataset, val_dataset, train_loader, val_loader, vis_loader
