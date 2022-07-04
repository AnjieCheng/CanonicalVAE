import yaml
import torch
from torchvision import transforms

from src import data
from src import canonicalvq


method_dict = {
    'canonicalvq': canonicalvq,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.
    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, out_dir, cfg, device):
    ''' Returns a trainer instance.
    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, out_dir, cfg, device)
    return trainer


# Generator for final output extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.
    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(cfg):
    ''' Returns the dataset.
    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    dataset_type = cfg['data']['dataset']

    # Create dataset
    if dataset_type == 'ShapeNet':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        train_dataset, val_dataset, train_loader, val_loader, vis_loader = data.build(cfg)

        h5_train_dataset = data.H5DataLoader(augment=False, choice=cfg['data']['cates'][0])
        h5_train_loader = torch.utils.data.DataLoader(h5_train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, 
                                                      num_workers=4,
                                                      drop_last=True,pin_memory=True)

        # h5_train_loader = data.H5DataLoader(augment=False, choice=cfg['data']['cates'][0])
        train_dataset = None
        train_loader = None
        # val_dataset = data.ShapeNetCore(cates=cfg['data']['cates'], split='val')

        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
        # vis_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

        # h5_train_dataset = data.ShapeNetCore(cates=cfg['data']['cates'], split='train')
        # h5_train_loader = torch.utils.data.DataLoader(h5_train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, 
        #                                               num_workers=int(4),
        #                                               drop_last=True,pin_memory=True)
        # h5_train_dataset = h5_train_loader = None

    elif dataset_type == 'ModelNet':
        raise NotImplementedError("Modelnet not supported yet")
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return train_dataset, val_dataset, train_loader, val_loader, vis_loader, h5_train_loader
