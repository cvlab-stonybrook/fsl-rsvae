import torch
import argparse
import numpy as np
from torch.autograd import Variable
from torchvision.datasets.folder import DatasetFolder
from torch.distributions import uniform, normal
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
import pdb
import yaml
import datasets.feature_loader as feat_loader
from sklearn.manifold import TSNE
import h5py
from scipy.stats import multivariate_normal
import scipy
def finetune_vae(feats_vae, x_shot, label_real):
    attributes = np.load('./mini_attr.npy')
    x_shot = x_shot.detach()
    z_dist = normal.Normal(0, 1)
    bs_list = np.arange(4)
    feats_vae.train()
    optimizer = torch.optim.Adam(feats_vae.parameters(), lr=0.0001)
    for ep in range(5):
      np.random.shuffle(bs_list)
      for idx in bs_list:
        targets = x_shot[idx]
        labels_sel = label_real[idx] + 80
        attr = torch.from_numpy(attributes[labels_sel]).float().cuda()
        attr = attr.repeat((1, 50)).reshape((5, 50, -1))
        Z = z_dist.sample((5, 50, 512)).cuda()
        concat_feats = torch.cat((Z, attr), dim=2)
        concat_feats = torch.autograd.Variable(concat_feats, requires_grad=True)
        feats = feats_vae.model(concat_feats).reshape((-1, 512))
        feats = feats_vae.relu(feats_vae.bn1(feats)).reshape((5, 50, 512))
        feats = feats.mean(1)
        feats = F.normalize(feats, dim=-1)
        mse_loss = F.mse_loss(feats, targets)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        print(mse_loss.item())



def shrink_feats(cl_data_file):
    cl_mean_file = {}
    weight_data_file = {}
    for k, v in cl_data_file.items():
        mean_feats = np.mean(v, 0)
        cl_mean_file[k] = mean_feats / np.sqrt(np.sum(mean_feats*mean_feats))
    for k, v in cl_data_file.items():
        v = np.array(v)
        v = v / np.sqrt(np.sum(v*v, -1, keepdims=True))
        dist = np.sum((v - cl_mean_file[k])**2, 1)
        sort_idx = np.argsort(dist)
        in_idx = sort_idx[:50]
        out_idx = sort_idx[50:200]
        in_feats = v[in_idx]
        out_feats = v[out_idx]
        cl_data_file[k] = []
        for in_f in in_feats:
          cl_data_file[k].append(in_f)
        for o_idx in out_idx:
          close_feats = (v[o_idx] + cl_mean_file[k]) / 2
          close_dist = np.sum((in_feats - close_feats)**2, -1)
          min_idx = np.argsort(close_dist)[0]
          cl_data_file[k].append(in_feats[min_idx])
    pdb.set_trace()
    return cl_data_file

def det(matrix):
    order=len(matrix)
    posdet=0
    for i in range(order):
        posdet+=reduce((lambda x, y: x * y), [matrix[(i+j)%order][j] for j in range(order)])
    negdet=0
    for i in range(order):
        negdet+=reduce((lambda x, y: x * y), [matrix[(order-i-j)%order][j] for j in range(order)])
    return posdet-negdet


def remove_feats(cl_data_file):
    cl_mean_file = {}
    cl_var_file = {}
    weight_data_file = {}
    remove_num = []
    for k, v in cl_data_file.items():
        mean_feats = np.mean(v, 0)
        cl_mean_file[k] = mean_feats
        cl_var_file[k] = np.cov(np.array(v).T)
    for k, v in cl_data_file.items():
        v = np.array(v)
        #dist = np.sum((v - cl_mean_file[k])**2, 1)
        #sort_idx = np.argsort(dist)[:220]
        cl_data_file[k] = []
        #weight_data_file[k] = []
        inv_var = scipy.linalg.inv(cl_var_file[k])
        mean = cl_mean_file[k]
        prob = np.sum(np.matmul((v-mean),inv_var)*(v-mean), -1)
        prob = 1-scipy.stats.chi2.cdf(prob, 512)
        #rv = multivariate_normal(mean = cl_mean_file[k], cov = cl_var_file[k])
        for idx in range(600):
          if prob[idx] > 0.9:
            cl_data_file[k].append(v[idx]) 
        remove_num.append(np.sum(prob<0.9))
        #for sidx in sort_idx:
        #  cl_data_file[k].append(v[sidx])
        #  cl_data_file[k].append((cl_mean_file[k] + (np.random.normal(v[sidx].shape)*0.001)).astype(np.float32))
        #  weight_data_file[k].append(1./dist[sidx])
        #all_feats = (np.random.multivariate_normal(mean=cl_mean_file[k], cov=cl_var_file[k], size=200)).astype(np.float32)
        #for all_feat in all_feats:
        #  cl_data_file[k].append(all_feat*0.3 + cl_mean_file[k]*0.7)
    pdb.set_trace()
    return cl_data_file

def interpolate_feats(cl_data_file):
    cl_mean_file = {}
    for k, v in cl_data_file.items():
        mean_feats = np.mean(v, 0)
        cl_mean_file[k] = mean_feats
    for k, v in cl_data_file.items():
        v = np.array(v) 
        dist = np.sum((v - cl_mean_file[k])**2, 1)
        cl_data_file[k] = []
        for iv in v:
          cl_data_file[k].append(0.6*iv+0.4*cl_mean_file[k])

    return cl_data_file

def get_vae_center(out_dir, split='train', use_mean=True):
    attr_out_file = os.path.join(out_dir, '%s_attr.hdf5'%split)
    vae_data_file = feat_loader.init_loader(attr_out_file)
    if 'train' in split:
      num = 64
    else:
      num = 20
    if use_mean:
      vae_feats_all = torch.zeros((num, 512))
    else:
      vae_feats_all = torch.zeros((num, 20, 512))
    mean_data_file = {}

    for k, feats in vae_data_file.items():
        mean_feats = np.mean(feats, 0)
        mean_data_file[k] = mean_feats

    for k, feats in vae_data_file.items():
        #mean_feats = np.array(feats)[:50]
        #mean_feats = 3*mean_feats - 2*mean_data_file[k]
        if use_mean:
          mean_feats = np.mean(feats, 0)
        else:
          mean_feats = np.array(feats)[:20]
        mean_feats = torch.from_numpy(mean_feats)
        mean_feats = F.normalize(mean_feats, dim=-1) 
        if 'test' in split: 
          k = k - 80
        vae_feats_all[k] = mean_feats
  
    return vae_feats_all

class FeatsVAE(nn.Module):
    def __init__(self, x_dim, latent_dim):
        super(FeatsVAE, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(x_dim+latent_dim, 4096),
            #nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            nn.LeakyReLU())
        self.linear_mu =  nn.Sequential(
            nn.Linear(4096, latent_dim),
            nn.ReLU())
        self.linear_logvar =  nn.Sequential(
            nn.Linear(4096, latent_dim),
            nn.ReLU())
        self.model = nn.Sequential(
            nn.Linear(2*latent_dim, 4096),
            nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            #nn.LeakyReLU(),
            nn.Linear(4096, x_dim),
            #nn.Sigmoid(),
        )
        self.bn1 = nn.BatchNorm1d(x_dim)
        self.relu = nn.ReLU(inplace=True)
        self.z_dist = normal.Normal(0, 1)
        self.init_weights()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        # remove abnormal points
        return mu + eps*std

    def init_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
              m.weight.data.normal_(0, 0.02)
              m.bias.data.normal_(0, 0.02)

    def forward(self, x, attr):
        x = torch.cat((x, attr), dim=1)
        x = self.linear(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        latent_feats = self.reparameterize(mu, logvar)
        #Z = self.z_dist.sample(attr.shape).cuda() 
        concat_feats = torch.cat((latent_feats, attr), dim=1)
        recon_feats = self.model(concat_feats)
        recon_feats = self.relu(self.bn1(recon_feats))
        return mu, logvar, recon_feats


class FeatureDataset(DatasetFolder):
    """Face Landmarks dataset."""

    def __init__(self, feature_dict, label_list=[]):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_list = label_list
        self.features, self.labels = self.convert_dict_to_list(feature_dict)

    def convert_dict_to_list(self, feature_dict):
        features = []
        labels = []
        for k, v in feature_dict.items():
            if k in self.label_list:
              continue
            features += v
            labels += [k] * len(v)
        return features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def generate_feats(feats_vae, attributes, output_file, label_list):
    f = h5py.File(output_file, 'w')
    ind_count = 500
    max_count = ind_count * len(label_list)
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    feats_vae.eval()
    z_dist = normal.Normal(0, 1)
    for label in label_list:
        attr = torch.from_numpy(attributes[label]).float().cuda()
        attr = attr.repeat(ind_count, 1)
        Z = z_dist.sample((ind_count, 512)).cuda()
        concat_feats = torch.cat((Z, attr), dim=1)
        feats = feats_vae.model(concat_feats)
        feats = feats_vae.relu(feats_vae.bn1(feats))
        if all_feats is None:      
          all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = np.array([label]*ind_count)
        count = count + feats.size(0)
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

       

def train_vae(feature_loader, feats_vae, attributes):
    optimizer = torch.optim.Adam(feats_vae.parameters(), lr=0.001)
    #for ep in range(10):
    for ep in range(60):
      loss_recon_all = 0
      loss_kl_all = 0
      for idx, (data, label) in enumerate(feature_loader):
        data = data.cuda()
        #weight = weight.cuda() / torch.sum(weight)
        attr = torch.from_numpy(attributes[label]).float().cuda()
        mu, logvar, recon_feats = feats_vae(data, attr)
        recon_loss = ((recon_feats - data)**2).mean(1)
        recon_loss = torch.mean(recon_loss)
        #kl_loss = -0.5*torch.sum(1+logvar-logvar.exp()-mu.pow(2)) / data.shape[0]
        kl_loss = (1+logvar-logvar.exp()-mu.pow(2)).sum(1)
        kl_loss = -0.5*torch.mean(kl_loss)
        L_vae = recon_loss+kl_loss*0.005
        optimizer.zero_grad()
        L_vae.backward()   
        optimizer.step()
        loss_recon_all += recon_loss.item()
        loss_kl_all += kl_loss.item()
      print('Ep: %d   Recon Loss: %f   KL Loss: %f'%(ep, loss_recon_all/(idx+1), loss_kl_all/(idx+1)))
    return feats_vae
    #torch.save({'state': feats_vae.state_dict()}, 'feats_vae_mini.pth') 


def visualize_feats(feats_dir):
    visual_feats = []
    attr_feats = []
    visual_labels = []
    attr_labels = []
    cl_data_file = os.path.join(feats_dir, 'test.hdf5')
    cl_data_file = feat_loader.init_loader(cl_data_file)
    vae_data_file = os.path.join(feats_dir, 'test_attr_ood.hdf5')
    vae_data_file = feat_loader.init_loader(vae_data_file)
    pdb.set_trace()
    #labels = [51, 3, 179, 7, 11, 175] 
    #labels = [15,6,17,8,9]
    labels = [85, 86, 87, 88, 89]
    #labels = [13, 17, 21, 29, 33, 37]
    tsne = TSNE(n_components=2, random_state=0)
    for idx in range(5):
        label = labels[idx]
        visual_feats.extend(cl_data_file[label-80][:100])
        #attr_feats.extend((vae_data_file[label][:300]))
        #attr_feats.extend(np.mean(np.array(vae_data_file[label]), 0, keepdims=True))
        visual_labels.extend([idx]*len(cl_data_file[label-80][:100]))
        #attr_labels.extend([idx]*len(vae_data_file[label][:300]))
        #attr_labels.extend([idx])
    visual_feats = np.array(visual_feats)
    #attr_feats = np.array(attr_feats)
    #all_feats = np.concatenate((visual_feats, attr_feats), 0)
    pdb.set_trace()
    all_labels = visual_labels 
    all_feats = visual_feats
    all_feats_2D = tsne.fit_transform(all_feats)
    #all_feats_2D = tsne.fit_transform(visual_feats)
    #all_labels = visual_labels
    colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k',  'orange', 'purple'])      
    for idx in range(all_feats_2D.shape[0]):
        feat = all_feats_2D[idx]
        #if feat[0] < -30 or feat[1] < -30:
        #  continue
        label = all_labels[idx]
        color = colors[label]
        if idx < visual_feats.shape[0]:
          marker = '*'
          #continue
        else:
          marker = 'o'
          #continue
        plt.scatter(feat[0], feat[1], c=color, marker=marker) 
    plt.savefig('features_base_mini.png')

def save_vae_features(out_file, attr_out_dir):
    cl_data_file = feat_loader.init_loader(out_file)
    cl_data_file = remove_feats(cl_data_file)
    feature_dataset = FeatureDataset(cl_data_file)
    feature_loader = torch.utils.data.DataLoader(feature_dataset, shuffle=True, pin_memory=True, drop_last=False, batch_size=256) 
    attributes = np.load('./mini_attr.npy')
    feats_vae = FeatsVAE(512, 512).cuda()
    feats_vae = train_vae(feature_loader, feats_vae, attributes)
    #feats_vae.load_state_dict(torch.load('feats_vae_mini.pth')['state'])
    #torch.save({'state': feats_vae.state_dict()}, 'feats_vae_mini.pth') 
    generate_feats(feats_vae, attributes, os.path.join(attr_out_dir, 'train_attr.hdf5'), np.arange(0, 64))
    generate_feats(feats_vae, attributes, os.path.join(attr_out_dir, 'test_attr.hdf5'), np.arange(80, 100))
    #return feats_vae



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    out_dir = os.path.dirname(config['load_encoder'])
    out_file = os.path.join(out_dir, 'features', 'test.hdf5')
    cl_data_file = feat_loader.init_loader(out_file)
    feature_dataset = FeatureDataset(cl_data_file)
    feature_loader = torch.utils.data.DataLoader(feature_dataset, shuffle=True, pin_memory=True, drop_last=False, batch_size=256)
     
    #attributes = np.load('./mini_attr.npy')
    #feats_vae = FeatsVAE(512, 512).cuda()
    #train_vae(feature_loader, feats_vae, attributes)
    #generate_feats(feats_vae, attributes, os.path.join(out_dir, 'features', 'test_attr.hdf5'), np.arange(80, 100))
    #save_vae_features(out_file, out_dir)
    vae_feats_file = os.path.join(out_dir, 'features', 'test_attr.hdf5')
    vae_data_file = feat_loader.init_loader(vae_feats_file)
    visualize_feats(cl_data_file, vae_data_file)
