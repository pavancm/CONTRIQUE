import numpy as np
import argparse
from sklearn.linear_model import Ridge
import pickle

def main(args):
    
    feat = np.load(args.feat_path)
    scores = np.load(args.ground_truth_path)
    
    #train regression
    reg = Ridge(alpha=args.alpha).fit(feat, scores)
    pickle.dump(reg, open('lin_regressor.save','wb'))

def parse_args():
    parser = argparse.ArgumentParser(description="linear regressor")
    parser.add_argument('--feat_path', type=str, help = 'path to features file')
    parser.add_argument('--ground_truth_path', type=str, \
                        help = 'path to ground truth scores')
    parser.add_argument('--alpha', type = float, default = 0.1, \
                        help = 'regularization coefficient')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)