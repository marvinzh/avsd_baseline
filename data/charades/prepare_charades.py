#!/usr/bin/env python
"""Image feature extration using a caffe model
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import argparse
import logging
import sys
import time
import os
import re
import glob
import json
import six

import numpy as np

import pickle
from scipy.misc import imread, imresize, imsave
import skimage.transform
import scipy.io as sio
from sklearn import preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meaningless warning
    # logging.warn("The value of --feature-dim is ignored. The feature will be saved in its original shape.")

    # output immediately
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    parser = argparse.ArgumentParser(description="Processing feature file into a signle file for training")
    parser.add_argument('--idmap',"-m",default='dict_charades_img2vid_mapping.json', type=str,help="image to video name translation mapping file")
    parser.add_argument('--feature-dir',"-f",default='i3d_rgb',
                        type=str, help="feature folder")
    parser.add_argument('--output',"-o", default='',
                        type=str, help="output feature name")
    parser.add_argument('--skip', default='1',
                        type=int, help="# of skip frame when downsampling")
    args = parser.parse_args()
    args.output='{}_features_charades.pkl'.format(args.output)
    
    # unused variable
    args.offset = 3
    # unused variable
    args.id_pattern = '/([^/\s]+)/jpg'
    if args.idmap != '':
        print("loading image-id mapping: {}".format(args.idmap))
        idmap = json.load(open(args.idmap, 'r'))
    
    # unused variable
    idpat = re.compile(args.id_pattern)

    # initial output dict
    output = {}
    for fname in os.listdir(args.feature_dir):
	# get filename
        vid = fname[0:5]
        if vid in idmap:
            print("translate {} into {}".format(vid, idmap[vid]))
            vid = idmap[vid]
        else:
            raise RuntimeError('Unknown Video ID ' + vid)
        y_feature = np.load(open(args.feature_dir + '/' + fname, 'r'))
        feature_downsampled = y_feature[::args.skip]
        print("step size: {}, original -> downsampled: {} -> {}, {} ,images in {}".format(args.skip,y_feature.shape[0],feature_downsampled.shape[0],vid,fname))
        #print('ID:', vid, '-- processing', y_feature.shape[0], 'images in', fname)
        output[vid] = feature_downsampled

    print("saving features to {}".format(args.output))
    pickle.dump(output, open(args.output, 'wb'), 2)
    print("done")
