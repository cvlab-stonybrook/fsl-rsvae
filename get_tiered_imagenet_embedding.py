
import clip
import numpy as np
model, preprocess = clip.load("ViT-B/32")
import torch
from tqdm.notebook import tqdm
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

#print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")
import pdb

def build_label_dict():
    name_dict1 = {}
    name_dict2 = {}
    name_files = open('./materials/tiered-imagenet/').readlines()
    for name_line in name_files:
      label, name = name_line.strip().split()
      name_dict1[name] = label
      name_dict2[label] = name
    return name_dict1, name_dict2
    
def zeroshot_classifier(templates, name_dict):
    attr_dict = {}
    with torch.no_grad():
        zeroshot_weights = []
        #for classname in (classnames):
        for classname in name_dict.keys():
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            attr_dict[classname] = class_embedding
    return attr_dict        
import pickle
def build_matrix(name_dict, attr_dict):
    attr_matrix = np.zeros((100, 512))
    for split in ['train_phase_train', 'train_phase_val', 'test']:
      pack_file = './materials/mini-imagenet/miniImageNet_category_split_%s.pickle' % split
      with open(pack_file, 'rb') as f:
        pack = pickle.load(f, encoding='latin1')
      for k, v in (pack['catname2label']).items():
        semantic_label = name_dict[k]
        attr = attr_dict[semantic_label]
        attr_matrix[v] = attr

    pdb.set_trace()
   
import json
name_dict1, name_dict2 = build_label_dict()
pdb.set_trace()
#attr_dict = zeroshot_classifier( imagenet_templates, name_dict1)
from scipy.io import loadmat
attr_dict = loadmat('mini_attr.mat')
attr_matrix = build_matrix(name_dict2, attr_dict)
pdb.set_trace()
