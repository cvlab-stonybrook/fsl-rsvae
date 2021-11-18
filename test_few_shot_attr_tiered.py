import argparse
import yaml
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os
import datasets
import models
import utils
import utils.few_shot as fs
import datasets.feature_loader as feat_loader
from datasets.samplers import CategoriesSampler
from extract_feats import save_features
from train_vae_tiered import save_vae_features, get_vae_center, visualize_feats, visualize_ood
from train_gan import save_gan_features, visualize_gan_feats
import copy
def get_augmented_feats(vae_feats, x_shot):
    iter_num = 0
    x_shot_aug = torch.zeros((4, 5, 10, 512))
    while True:
      loss = torch.sum((vae_feats - x_shot)**2)
      loss.backward()
      x_shot.data = x_shot.data - (1.0/1000.0) * (torch.sign(x_shot.grad.data))
      x_shot.grad.data.zero_() 
      iter_num += 1
      if iter_num % 10 == 0:
        aug_num = int(iter_num / 10) - 1
        x_shot_aug[:,:,aug_num,:] = copy.deepcopy(x_shot.detach())
      if iter_num == 100:
        break
    return torch.mean(x_shot_aug, 2).cuda()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], split='train')
    #train_dataset = datasets.make(config['dataset'], split='train')
    #utils.log('dataset: {} (x{}), {}'.format(
    #        dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 4
    #batch_sampler = CategoriesSampler(
    #        dataset.label, n_batch, n_way, n_shot + n_query,
    #        ep_per_batch=ep_per_batch)
    #loader = DataLoader(dataset, batch_sampler=batch_sampler,
    #                    num_workers=8, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False,
    #                    num_workers=8, pin_memory=True)

    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    torch.cuda.manual_seed(666)
    va_lst = []
    out_dir = os.path.dirname(config['load_encoder'])
    out_file = os.path.join(out_dir, 'features', 'train.hdf5')
    feats_dir = os.path.join(out_dir, 'features')
    print('Save training set features ...')
    #save_features(model, train_loader, out_file)
    #visualize_ood(feats_dir)
    #pdb.set_trace()
    print('Trainig CVAE ...')
    visualize_ood(feats_dir)
    pdb.set_trace()
    #save_vae_features(out_file, feats_dir)
    vae_center = get_vae_center(feats_dir, split='test')
    for epoch in range(1, test_epochs + 1):
        for data, label in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)
            with torch.no_grad():
            #if True:
                x_shot, x_query, metric = model(x_shot, x_query)
                x_shot = x_shot.detach()
                x_query = x_query.detach()
                label_real = label.view(ep_per_batch, n_way, -1)[:,:,0].long()
                vae_feats = vae_center[label_real].cuda()
                #dist = torch.sqrt(torch.sum((x_shot-vae_feats.unsqueeze(2))**2, -1))
                #dist = torch.exp(-0.5 * dist)
                #dist = dist / torch.sum(dist, -1, keepdims=True)
                #vae_feats = x_shot + torch.rand(x_shot.shape).cuda() * 0.001
                #x_shot_aug = torch.autograd.Variable(copy.deepcopy(x_shot.detach()), requires_grad=True)
                #aug_feats = get_augmented_feats(vae_feats, x_shot_aug)
                #x_shot = x_shot*((n_shot)/(n_shot+3)) + vae_feats*(2/(n_shot+3))
                #x_shot = torch.sum(x_shot*dist.unsqueeze(-1), -2)
                x_shot = torch.mean(x_shot, -2)
                x_shot = x_shot*((n_shot)/(n_shot+1)) + vae_feats*(1/(n_shot+1))
                #x_shot = vae_feats
                x_shot = F.normalize(x_shot, dim=-1) 
                logits = utils.compute_logits(
                       x_query, x_shot, metric=metric, temp=model.temp).view(-1, n_way)
                ep_label = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                acc = utils.compute_acc(logits, ep_label)
                aves['va'].add(acc, len(data))
                va_lst.append(acc)
        print('test epoch {}: acc={:.2f} +- {:.2f} (%) (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100, label[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)

