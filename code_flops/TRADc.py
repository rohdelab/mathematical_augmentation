#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import multiprocessing as mp
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils import *
from skimage.feature import hog
from skimage import data, exposure
from tqdm import tqdm
from PIL import Image
#from subspace_classifier import NS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import cv2
from sklearn.cluster import KMeans
import time
import pywt
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--feature', type=str, required=True, choices=['raw', 'hog', 'sift', 'wavelet'])
parser.add_argument('--classifier', required=True, choices=['KNN', 'SVM', 'subspace', 'rbf-svm', 'LR'])
parser.add_argument('--naug', default=1, type=int)
args = parser.parse_args()

num_classes, img_size, po_train_max, rm_edge = dataset_config(args.dataset)

eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]
Rdown = 4  # downsample radon projections (w.r.t. angles)
theta = np.linspace(0, 176, 180 // Rdown)

def extract_wavelet(imgs):
    fd = []
    for img in imgs:
        coeffs2 = pywt.dwt2(img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fd.append(np.concatenate([LL.ravel(), LH.ravel(), HL.ravel(), HH.ravel()]))
    fd = np.array(fd)
    return fd


def extract_sift(imgs):
    fd = []
    for img in imgs:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        if des is None:
            fd.append(np.zeros((1, 128), dtype=np.float32))
        else:
            fd.append(des.astype(np.float32))
    fd = np.array(fd)
    return fd

def extract_sift_parallel(imgs):
    n_cpu = np.min([mp.cpu_count(), imgs.shape[0]])
    splits = np.array_split(imgs, n_cpu, axis=0)
    pl = mp.Pool(n_cpu)

    fd = pl.map(extract_sift, splits)
    fd = np.array([img_fds for batch_img_fds in fd for img_fds in batch_img_fds])
    pl.close()
    pl.join()
    return fd

def bag_of_words_encode(kmeans, img_fd):
    img_fd_clusters = kmeans.predict(img_fd)
    code = np.bincount(img_fd_clusters, minlength=kmeans.n_clusters)
    return code

def extract_hog(imgs):
    fd = np.array([hog(img) for img in imgs]).astype(np.float32)
    return fd

def extract_hog_parallel(imgs):
    n_cpu = np.min([mp.cpu_count(), imgs.shape[0]])
    splits = np.array_split(imgs, n_cpu, axis=0)
    pl = mp.Pool(n_cpu)

    fd = pl.map(extract_hog, splits)
    fd = np.vstack(fd)  # (n_samples, features)
    pl.close()
    pl.join()
    return fd


if __name__ == '__main__':
    datadir = '../data'
    # x_train: (n_samples, width, height)
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset, num_classes, datadir)
    

    if args.feature != 'raw':
        cache_file = os.path.join(datadir, args.dataset, f'{args.feature}.hdf5')
        print(cache_file)
        if os.path.exists(cache_file):
            with h5py.File(cache_file, 'r') as f:
                x_train, y_train = f['x_train'][()], f['y_train'][()]
                x_test, y_test = f['x_test'][()], f['y_test'][()]
                x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
                print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
        else:
            if args.feature == 'hog':
                # x_train, y_train = x_train[:10], y_train[:10]
                print('computing hog...')
                x_train = extract_hog_parallel(x_train[..., np.newaxis])
                x_test = extract_hog_parallel(x_test[..., np.newaxis])
            elif args.feature == 'sift':
                train_img_fds = extract_sift_parallel(x_train)
                test_img_fds = extract_sift_parallel(x_test)
                all_train_fds = np.array([fd for img_fds in train_img_fds for fd in img_fds])
                kmeans = KMeans(n_clusters=10, random_state=0).fit(all_train_fds)
                train_codes = np.array([bag_of_words_encode(kmeans, img_fd) for img_fd in train_img_fds])
                test_codes = np.array([bag_of_words_encode(kmeans, img_fd) for img_fd in test_img_fds])
                x_train, x_test = train_codes, test_codes
            elif args.feature == 'wavelet':
                train_img_fds = extract_wavelet(x_train)
                test_img_fds = extract_wavelet(x_test)
                pca = PCA(n_components=100)
                x_train = pca.fit_transform(train_img_fds)
                x_test = pca.transform(test_img_fds)
            with h5py.File(cache_file, 'w') as f:
                x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
                f.create_dataset('x_train', data=x_train)
                f.create_dataset('y_train', data=y_train)
                f.create_dataset('x_test', data=x_test)
                f.create_dataset('y_test', data=y_test)
                print('saved to {}'.format(cache_file))

    num_repeats = 1
    accs = []
    all_preds = []
    AUG_N = args.naug
    

    x_train = (x_train.astype(np.float32) / 255. - 0.5) / 0.5
    x_test = (x_test.astype(np.float32) / 255. - 0.5) / 0.5
    

    x_test_Rearrange = x_test #rearrange(x_test, 'b c w h -> b w h c') 
    x_test_BeforeRearrange = [] 
    for eachTest in x_test_Rearrange:
        x_test_BeforeRearrange.append(np.pad(eachTest, ((15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(15+int(np.round(img_size/2)),15+int(np.round(img_size/2)))), 'constant', constant_values=-1)) # pad the original image ((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array) 
    x_test = np.asarray(x_test_BeforeRearrange) # rearrange(np.asarray(x_test_BeforeRearrange), 'b w h c -> b c w h') 
    
    
    for n_samples_perclass in [2**i for i in range(0, po_train_max+1)]:
        for repeat in range(num_repeats):
            x_train_sub_beforeAug, y_train_sub_beforeAug = take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            
            
            x_train_sub_beforeAugReshape = x_train_sub_beforeAug #rearrange(x_train_sub_beforeAug, 'b c w h -> b w h c') 
            np.random.seed(seed=None)
            # x_train_sub_breforeReshape, y_train_sub_aug = take_affine_tform_aug(x_train_sub_beforeAugReshape, y_train_sub_beforeAug,img_size,args.dataset, N_aug=AUG_N) #
            x_train_sub_breforeReshape, y_train_sub_aug = take_affine_tform_augV2trad(x_train_sub_beforeAugReshape, y_train_sub_beforeAug,img_size,args.dataset, N_aug=AUG_N)
            np.random.seed(0)
            x_train_sub_aug = x_train_sub_breforeReshape # rearrange(x_train_sub_breforeReshape, 'b w h c -> b c w h') 
            x_train_sub_beforeAugPad = []
            for eachX_trainBeforePad in x_train_sub_beforeAugReshape:
                x_train_sub_beforeAugPad.append(np.pad(eachX_trainBeforePad, ((15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(15+int(np.round(img_size/2)),15+int(np.round(img_size/2)))), 'constant', constant_values=-1)) # pad the original image ((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array)
            x_train_original = np.asarray(x_train_sub_beforeAugPad) # rearrange(np.asarray(x_train_sub_beforeAugPad), 'b w h c -> b c w h') 
            x_train_sub = np.concatenate((x_train_original, x_train_sub_aug), axis=0)
            y_train_sub = np.concatenate((y_train_sub_beforeAug, y_train_sub_aug), axis=0)
            
            # from scipy.io import savemat
            # mdic = {"Xtr0": x_train_sub_beforeAug,"Xtr": x_train_sub,"Xtr_aug": x_train_sub_aug}
            # # mdic = {"Xtr0": x_train_sub_beforeAug,"Xtr": x_train_sub,"Xtr_aug": x_train_sub_aug,"Xte":x_test}
            # savemat("hey_shifat.mat", mdic)
            # print(asdasdas)
            
            pca=PCA(n_components=min(x_train_sub.shape[0],x_train_sub.shape[1]*x_train_sub.shape[2],128))

            # x_train_sub_flat = x_train_sub.reshape(x_train_sub.shape[0], -1)
            # x_test_flat = x_test.reshape(x_test.shape[0], -1)
          
            x_train_sub_flat = pca.fit_transform(x_train_sub.reshape(x_train_sub.shape[0], -1))
            x_test_flat = pca.transform(x_test.reshape(x_test.shape[0], -1))

            
            if args.classifier == 'subspace':
                clf = NS(num_classes=num_classes)
            elif args.classifier == 'LR':
                clf = LogisticRegression()
            elif args.classifier == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=3)
            elif args.classifier == 'SVM':
                clf = LinearSVC()
            elif args.classifier == 'rbf-svm':
                clf = SVC(kernel='rbf')
            
            clf.fit(x_train_sub_flat, y_train_sub)
            preds = clf.predict(x_test_flat)
            # print('Runtime of fit+predict functions: {} seconds'.format(toc-tic))
            accs.append(accuracy_score(y_test, preds))
            all_preds.append(preds)
            print('n_samples_perclass {} repeat {} acc {}'.format(n_samples_perclass, repeat, accuracy_score(y_test, preds)))

    accs = np.array(accs).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])

    results_dir = 'results/final/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'trad_aug'+str(AUG_N)+'_{}.hdf5'.format(args.classifier))
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)
    print(f'saved {result_file}')
