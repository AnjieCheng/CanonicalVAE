import os
from src.canonicalvq import models, training #, generation
from src import config, data

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Unfold Network model.
    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    feat_dim = cfg['model']['feat_dim']

    encoder = models.encoder_dict['DGCNN'](feat_dim=feat_dim)

    # Fold = models.decoder_dict['FixPointGenerator'](z_dim=256)
    
    if cfg['model']['stage1']['decoder_f'] == 'FOLDINGNET':
        Fold = models.decoder_dict['ImplicitFun'](z_dim=feat_dim, add_dim=3)
    elif cfg['model']['stage1']['decoder_f'] == 'SPGAN':
        Fold = models.decoder_dict['SPGANGenerator'](z_dim=feat_dim, add_dim=3, use_local=cfg['model']['stage1']['SPGAN_f_use_local'])

    if cfg['model']['stage1']['decoder_u'] == 'FOLDINGNET':
        Unfold = models.decoder_dict['ImplicitFun'](z_dim=feat_dim, add_dim=3)
    elif cfg['model']['stage1']['decoder_u'] == 'SPGAN':
        Unfold = models.decoder_dict['SPGANGenerator'](z_dim=feat_dim, add_dim=3, use_local=cfg['model']['stage1']['SPGAN_u_use_local'])
    
    if cfg['model']['stage1']['decoder_f'] == 'FOLDINGNET':
        Fold_f = models.decoder_dict['ImplicitFun'](z_dim=feat_dim, add_dim=3)
    else:
        Fold_f = models.decoder_dict['SPGANGenerator'](z_dim=feat_dim, add_dim=3, use_local=True)

    Pg = models.decoder_dict['PrimitiveGrouping'](num_groups=cfg['model']['stage2']['num_groups'])

    vocab_size = cfg['model']['stage2']['vocab_size']
    Vq = models.encoder.VectorQuantizerEMA(vocab_size, feat_dim, 0.25, 0.99) # num_embeddings, embedding_dim, commitment_cost, decay vocab_size
    # Vq = models.encoder.VectorQuantizer2(n_e=vocab_size, e_dim=128, beta=0.25)
    # Vq = models.encoder.GumbelQuantize(128, 128, n_embed=5000, kl_weight=1e-8, temp_init=1.0)

    if cfg['model']['stage2']['discriminator']:
        D = models.decoder.Discriminator()
    else:
        D = None

    if cfg['model']['stage2']['perceptual']:
        P = models.decoder.PerceptualLoss()
    else:
        P = None

    if cfg['training']['mode'] == "stage3":
        transformer = models.transformer.mingpt.GPT(vocab_size=vocab_size, 
                                                    block_size=256,
                                                    n_layer=cfg['model']['transformer']['n_layer'], # 24
                                                    n_head=cfg['model']['transformer']['n_head'], # 16
                                                    n_embd=cfg['model']['transformer']['n_embd']) # 512
    else:
        transformer = None


    model = models.CanonicalVQ(encoder=encoder, 
                               fold=Fold, 
                               unfold=Unfold, 
                               vq=Vq, 
                               transformer=transformer,
                               fold_f=Fold_f, 
                               pg=Pg,
                               D=D,
                               P=P,
                               device=device)

    return model

def get_trainer(model, out_dir, cfg, device, **kwargs):
    ''' Returns the trainer object.
    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    # out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.Trainer(
        model,
        device=device,
        vis_dir=vis_dir,
        config=cfg,
    )

    return trainer

