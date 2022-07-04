import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import torch
from functools import partial
from src.utils.visualize import Visualizer
from external.metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD, emd_cd
from src.data.point_operation import normalize_point_cloud

class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader, epoch_it, save=None):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        all_gen = list()
        all_rec = list()
        all_ref = list()
        others = {}

        self.tmp_rec = None
        self.tmp_gen = None

        for data in tqdm(val_loader):
            raw_data = self.eval_step(data, epoch_it)

            for k in raw_data.keys():
                if k == "gen_shape":
                    all_gen.append(raw_data[k])
                elif k == "gdt_shape":
                    all_ref.append(raw_data[k])
                elif k == "rec_shape":
                    all_rec.append(raw_data[k])
                else:
                    if k in others:
                        others[k].append(raw_data[k])
                    else:
                        others[k] = list()
                        others[k].append(raw_data[k])

            torch.cuda.empty_cache()

        results = {}
        all_ref = torch.cat(all_ref, 0)

        if self.config['evaluate']['generation']:
            all_gen = torch.cat(all_gen, 0) 

            # all_gen = torch.tensor(normalize_point_cloud(all_gen.cpu().numpy())).cuda()
            # all_ref = torch.tensor(normalize_point_cloud(all_ref.cpu().numpy())).cuda()

            import pdb; pdb.set_trace()
            print("Generation Evaluation start")
            print(f"Generation Sample size:{all_gen.size()} Ref size: {all_ref.size()}")

            results = compute_all_metrics(all_gen, all_ref, 256, accelerated_cd=True)
            results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}

            sample_pcl_npy = all_gen.cpu().detach().numpy()
            ref_pcl_npy = all_ref.cpu().detach().numpy()

            jsd = JSD(sample_pcl_npy, ref_pcl_npy)
            results.update({'JSD': jsd})
            print(results)

            self.tmp_gen = all_gen.cpu().numpy()

        if self.config['evaluate']['reconstruction']:
            all_rec = torch.cat(all_rec, 0)
            print("Reconstruction Evaluation start")
            print(f"Reconstruction Sample size:{all_rec.size()} Ref size: {all_ref.size()}")

            emd_cd_rt = emd_cd(all_rec, all_ref, 128, accelerated_cd=True)
            emd_cd_rt = {
                k: (v.cpu().detach().item() if not isinstance(v, float) else v)
                for k, v in emd_cd_rt.items()}
            
            results.update(emd_cd_rt)

            for k in others.keys():
                all_other = torch.cat(others[k], 0)
                print("Reconstruction Evaluation (", k,") start")
                print(f"Reconstruction Other. Sample size:{all_other.size()} Ref size: {all_ref.size()}")

                emd_cd_rt = emd_cd(all_other, all_ref, 128, accelerated_cd=True)
                emd_cd_rt = {
                    k+'_'+j: (v.cpu().detach().item() if not isinstance(v, float) else v)
                    for j, v in emd_cd_rt.items()}
                
                results.update(emd_cd_rt)
            print(results)

            self.tmp_rec = all_rec.cpu().numpy()

        if save is not None:
            return 
            # save_file = os.path.join(save, str(epoch_it)+'_test.npy')
            # with open(save_file, 'wb') as f:
            #     np.save(f, raw_np_list)

            # save_file = os.path.join(save, 'best.npy')
            # with open(save_file, 'wb') as f:
            #     np.save(f, raw_np_list)
        return results

    def save_samples(self, out_dir, epoch_it):
        if self.tmp_rec is not None:
            save_file = os.path.join(out_dir, str(epoch_it)+'_rec.npy')
            with open(save_file, 'wb') as f:
                np.save(f, self.tmp_rec)
        if self.tmp_gen is not None:
            save_file = os.path.join(out_dir, str(epoch_it)+'_gen.npy')
            with open(save_file, 'wb') as f:
                np.save(f, self.tmp_gen)

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError

    def start_visdom(self):
        self.visualizer = Visualizer(self.config['visdom']['port'], self.config['visdom']['env'])