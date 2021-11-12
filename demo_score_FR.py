import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np

import os
import argparse
import pickle

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # load reference image
    ref_image = Image.open(args.ref_path)
    
    # load distorted image
    dist_image = Image.open(args.dist_path)
    
    # downscale images by 2
    sz = ref_image.size
    ref_image_2 = ref_image.resize((sz[0] // 2, sz[1] // 2))
    dist_image_2 = dist_image.resize((sz[0] // 2, sz[1] // 2))
    
    # transform to tensor
    ref_image = transforms.ToTensor()(ref_image).unsqueeze(0).cuda()
    ref_image_2 = transforms.ToTensor()(ref_image_2).unsqueeze(0).cuda()
    
    dist_image = transforms.ToTensor()(dist_image).unsqueeze(0).cuda()
    dist_image_2 = transforms.ToTensor()(dist_image_2).unsqueeze(0).cuda()
    
    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)
    
    # extract features
    model.eval()
    with torch.no_grad():
        _,_, _, _, ref_feat, ref_feat_2, _, _ = model(ref_image, ref_image_2)
        _,_, _, _, dist_feat, dist_feat_2, _, _ = model(dist_image, dist_image_2)
    
    ref = np.hstack((ref_feat.detach().cpu().numpy(),\
                                ref_feat_2.detach().cpu().numpy()))
    dist = np.hstack((dist_feat.detach().cpu().numpy(),\
                                dist_feat_2.detach().cpu().numpy()))
    feat = np.abs(ref - dist)
    
    # load regressor model
    regressor = pickle.load(open(args.linear_regressor_path, 'rb'))
    score = regressor.predict(feat)[0]
    print(score)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ref_path', type=str, \
                        default='sample_images/womanhat.bmp', \
                        help='Path to reference image', metavar='')
    parser.add_argument('--dist_path', type=str, \
                        default='sample_images/img191.bmp', \
                        help='Path to distorted image', metavar='')
    parser.add_argument('--model_path', type=str, \
                        default='models/CONTRIQUE_checkpoint25.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--linear_regressor_path', type=str, \
                        default='models/LIVE_FR.save', \
                        help='Path to trained linear regressor', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)