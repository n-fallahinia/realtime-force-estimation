"""
functions for K fold cross validation of 
the training dataset
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

from __future__ import print_function
from data_util import *

import glob 
import os  
import sys
import shutil 
from pathlib import Path
import numpy as np
from PIL import Image  
import PIL 

def k_fold_split(K_folds, mu = 10):
    """
    split the traing dataset into K folds for validation

    Inputs:
    - K_folds: number of the folds (usually 5)
    - mu: fold size adjustment coeff
    """
    dataset_path = './dataset/'
    train_image_path = dataset_path + 'train/image/'
    train_force_path = dataset_path + 'train/forces/'
    folds_path = dataset_path + 'train/folds/'

    if (len(os.listdir(folds_path)) != 0):
        remove_old_flag = input ("Old folds exists, would you like to reomve them? [Y/N]: ") 
        if remove_old_flag == 'Y' or remove_old_flag == 'y':
            print('Deleting old folds ...')
            fold_2_rem = glob.glob(folds_path + '*')
            [shutil.rmtree(fold2rem) for fold2rem in fold_2_rem] 
        elif remove_old_flag == 'N' or remove_old_flag == 'n':
            print('Continuing with old folds...')
            return
        else:
            print('Invalid input !!')
            k_fold_split(K_folds)

    image_list = glob.glob(train_image_path + '*.jpg')
    force_list = load_force_txt(train_force_path + 'force.txt',len(image_list))

    mask = list(range(len(force_list)))
    fold_size = (len(force_list)// K_folds) + mu
    fold_masks = list(fold_chunks(mask,fold_size))
    print('There will be %d new folds with the size of %d'%(K_folds, fold_size))
    print('\tWARNING: last fold will be %d'% len(fold_masks[-1]))

    print('Creating new folds ...')
    for fold_Idx, mask in enumerate(fold_masks):
        new_fold = 'fold_'+f'{fold_Idx:02d}'
        path = os.path.join(folds_path, new_fold) 
        os.mkdir(path) 
        os.mkdir(path + '/forces/')
        os.mkdir(path + '/image/') 
        print('\t created fold_'+f'{fold_Idx:02d}')
        # saving data into the folds 
        write_to_fold(folds_path+'fold_'+f'{fold_Idx:02d}', force_list, image_list, mask)
    print('Done ..!')

def write_to_fold (path, force_list, image_list, mask):
    """write forces and images to each fold"""
    force_2_write = force_list[mask]
    image_2_write = [image_list[i] for i in mask] 

    with open(path +'/forces/force.txt', 'w') as filestream: 
        for force in force_2_write:
            filestream.write("%s,%s,%s\n"%(force[0], force[1], force[2]))
    print('\t\tforces saved!') 

    for Idx, train_image in enumerate(image_2_write):
        img = Image.open(train_image)
        img.save(path + '/image/img_'+f'{Idx:04d}.jpg')
        # if Idx%100 == 0:
        #     print('\t%d images are saved'% Idx);       
    print('\t\timages saved! ')


def fold_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __debug__:
    print('DEBUG')
    K_folds = 5
    k_fold_split(K_folds)